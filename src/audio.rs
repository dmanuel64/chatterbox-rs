use ndarray::{Array2, ArrayD};
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction, audioadapter_buffers::direct::SequentialSliceOfVecs,
};
use std::{io::Cursor, path::Path};
use symphonia::core::{
    codecs::{CodecParameters, audio::AudioDecoderOptions},
    formats::{FormatOptions, TrackType, probe::Hint},
    io::{MediaSourceStream, MediaSourceStreamOptions},
    meta::MetadataOptions,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("failed to decode reference audio: {0}")]
    Symphonia(#[from] symphonia::core::errors::Error),
    #[error("reference audio has no audio track")]
    NoAudioTrack,
    #[error("reference audio's codec did not report a sample rate")]
    UnknownSampleRate,
    #[error("failed to resample reference audio: {0}")]
    ResamplerConstruction(#[from] rubato::ResamplerConstructionError),
    #[error("failed to resample reference audio: {0}")]
    Resample(#[from] rubato::ResampleError),
    #[error("failed to write WAV file: {0}")]
    Wav(#[from] hound::Error),
}

/// Loads an audio file of any supported format into mono `f32` samples at `target_sample_rate`,
/// shaped `(1, num_samples)`.
///
/// Equivalent to `librosa.load(path, sr=target_sample_rate)` (mono downmix + resample) followed
/// by `audio_values[np.newaxis, :]`.
pub fn load(bytes: Vec<u8>, target_sample_rate: u32) -> Result<ArrayD<f32>, Error> {
    let (samples, sample_rate) = decode_to_mono_f32(bytes)?;
    let samples = if sample_rate == target_sample_rate {
        samples
    } else {
        resample_mono(samples, sample_rate, target_sample_rate)?
    };
    let len = samples.len();
    Ok(Array2::from_shape_vec((1, len), samples)
        .expect("sample count should match array shape")
        .into_dyn())
}

/// Writes mono `f32` samples to a WAV file at `output_path`, at `sample_rate`.
pub fn write(samples: &[f32], sample_rate: u32, output_path: impl AsRef<Path>) -> Result<(), Error> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(output_path, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

fn decode_to_mono_f32(bytes: Vec<u8>) -> Result<(Vec<f32>, u32), Error> {
    let source = Box::new(Cursor::new(bytes));
    let mss = MediaSourceStream::new(source, MediaSourceStreamOptions::default());
    let hint = Hint::new();
    let mut format = symphonia::default::get_probe().probe(
        &hint,
        mss,
        FormatOptions::default(),
        MetadataOptions::default(),
    )?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or(Error::NoAudioTrack)?;
    let track_id = track.id;
    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(CodecParameters::audio)
        .ok_or(Error::NoAudioTrack)?
        .clone();
    let sample_rate = audio_params.sample_rate.ok_or(Error::UnknownSampleRate)?;

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&audio_params, &AudioDecoderOptions::default())?;

    let mut mono_samples: Vec<f32> = Vec::new();
    let mut scratch: Vec<f32> = Vec::new();
    while let Some(packet) = format.next_packet()? {
        if packet.track_id != track_id {
            continue;
        }
        let decoded = decoder.decode(&packet)?;
        let channels = decoded.spec().channels().count();
        decoded.copy_to_vec_interleaved::<f32>(&mut scratch);

        if channels <= 1 {
            mono_samples.extend_from_slice(&scratch);
        } else {
            mono_samples.extend(
                scratch
                    .chunks_exact(channels)
                    .map(|frame| frame.iter().sum::<f32>() / channels as f32),
            );
        }
    }

    Ok((mono_samples, sample_rate))
}

fn resample_mono(samples: Vec<f32>, from_rate: u32, to_rate: u32) -> Result<Vec<f32>, Error> {
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: None,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = Async::<f32>::new_sinc(
        to_rate as f64 / from_rate as f64,
        2.0,
        &params,
        1024,
        1,
        FixedAsync::Input,
    )?;

    let input_len = samples.len();
    let input_data = vec![samples];
    let input = SequentialSliceOfVecs::new(&input_data, 1, input_len)
        .expect("input buffer length should match input_len");

    let output = resampler.process_all(&input, input_len, None)?;
    Ok(output.take_data())
}
