#![warn(rust_2018_idioms)]
#![allow(elided_lifetimes_in_paths)]
use num::Complex;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::time::Instant;

/// Try to determine if `c` is in the Mandelbrot set, using at most `limit`
/// iterations to decide.
///
/// If `c` is not a member, return `Some(i)`, where `i` is the number of
/// iterations it took for `c` to leave the circle of radius two centered on the
/// origin. If `c` seems to be a member (more precisely, if we reached the
/// iteration limit without being able to prove that `c` is not a member),
/// return `None`.
fn escape_time(c: Complex<f64>, limit: usize) -> Option<usize> {
    let mut z = Complex { re: 0.0, im: 0.0 };
    for i in 0..limit {
        if z.norm_sqr() > 4.0 {
            return Some(i);
        }
        z = z * z + c;
    }

    None
}

use std::str::FromStr;

/// Parse the string `s` as a coordinate pair, like `"400x600"` or `"1.0,0.5"`.
///
/// Specifically, `s` should have the form <left><sep><right>, where <sep> is
/// the character given by the `separator` argument, and <left> and <right> are both
/// strings that can be parsed by `T::from_str`.
///
/// If `s` has the proper form, return `Some<(x, y)>`. If it doesn't parse
/// correctly, return `None`.
fn parse_pair<T: FromStr>(s: &str, separator: char) -> Option<(T, T)> {
    // Trim the input string to remove any leading or trailing whitespace
    let s = s.trim_matches(|c| c == ' ' || c == '(' || c == ')');

    // Remove surrounding parentheses if they exist
    // let s = if s.starts_with('(') && s.ends_with(')') {
    //     &s[1..s.len() - 1]
    // } else {
    //     s
    // };
    match s.find(separator) {
        None => None,
        Some(index) => match (T::from_str(&s[..index]), T::from_str(&s[index + 1..])) {
            (Ok(l), Ok(r)) => Some((l, r)),
            _ => None,
        },
    }
}

#[test]
fn test_parse_pair() {
    assert_eq!(parse_pair::<i32>("", ','), None);
    assert_eq!(parse_pair::<i32>("10,", ','), None);
    assert_eq!(parse_pair::<i32>(",10", ','), None);
    assert_eq!(parse_pair::<i32>("10,20", ','), Some((10, 20)));
    assert_eq!(parse_pair::<i32>("10,20xy", ','), None);
    assert_eq!(parse_pair::<f64>("0.5x", 'x'), None);
    assert_eq!(parse_pair::<f64>("0.5x1.5", 'x'), Some((0.5, 1.5)));
}

/// Parse a pair of floating-point numbers separated by a comma as a complex
/// number.
fn parse_complex(s: &str) -> Option<Complex<f64>> {
    match parse_pair(s, ',') {
        Some((re, im)) => Some(Complex { re, im }),
        None => None,
    }
}

#[test]
fn test_parse_complex() {
    assert_eq!(
        parse_complex("1.25,-0.0625"),
        Some(Complex {
            re: 1.25,
            im: -0.0625
        })
    );
    assert_eq!(parse_complex(",-0.0625"), None);
}

/// Given the row and column of a pixel in the output image, return the
/// corresponding point on the complex plane.
///
/// `bounds` is a pair giving the width and height of the image in pixels.
/// `pixel` is a (column, row) pair indicating a particular pixel in that image.
/// The `upper_left` and `lower_right` parameters are points on the complex
/// plane designating the area our image covers.
fn pixel_to_point(
    bounds: (usize, usize),
    pixel: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) -> Complex<f64> {
    let (width, height) = (
        lower_right.re - upper_left.re,
        upper_left.im - lower_right.im,
    );
    Complex {
        re: upper_left.re + pixel.0 as f64 * width / bounds.0 as f64,
        im: upper_left.im - pixel.1 as f64 * height / bounds.1 as f64, // Why subtraction here? pixel.1 increases as we go down,
                                                                       // but the imaginary component increases as we go up.
    }
}

#[test]
fn test_pixel_to_point() {
    assert_eq!(
        pixel_to_point(
            (100, 200),
            (25, 175),
            Complex { re: -1.0, im: 1.0 },
            Complex { re: 1.0, im: -1.0 }
        ),
        Complex {
            re: -0.5,
            im: -0.75
        }
    );
}

fn render(
    pixels: &mut [u8],
    bounds: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) {
    assert!(pixels.len() == bounds.0 * bounds.1);

    for row in 0..bounds.1 {
        for column in 0..bounds.0 {
            let point = pixel_to_point(bounds, (column, row), upper_left, lower_right);
            pixels[row * bounds.0 + column] = match escape_time(point, 255) {
                None => 0,
                Some(count) => 255 - count as u8,
            };
        }
    }
}

use image::png::PNGEncoder;
use image::ColorType;
use std::fs::File;

/// Write the buffer `pixels`, whose dimensions are given by `bounds`, to the
/// file named `filename`.
fn write_image(
    filename: &str,
    pixels: &[u8],
    bounds: (usize, usize),
) -> Result<(), std::io::Error> {
    let output = File::create(filename)?;

    let encoder = PNGEncoder::new(output);
    encoder.encode(
        &pixels,
        bounds.0 as u32,
        bounds.1 as u32,
        ColorType::Gray(8),
    )?;

    Ok(())
}

use std::env;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "mandelbrot")]
struct Opt {
    /// Input file
    #[structopt(name = "FILE")]
    file: String,

    /// Image dimensions in the format WIDTHxHEIGHT
    #[structopt(name = "PIXELS")]
    pixels: String,

    /// Upper left corner point in the format RE,IM
    #[structopt(name = "UPPERLEFT")]
    upper_left: String,

    /// Lower right corner point in the format RE,IM
    #[structopt(name = "LOWERRIGHT")]
    lower_right: String,

    /// Number of threads to use
    #[structopt(short = "t", long = "threads", default_value = "8")]
    threads: usize,
    /// Parallelism mode, either "single", "bands", or "rayon"
    #[structopt(short = "p", long = "parallelism", default_value = "single", possible_values = &["single", "bands", "rayon"])]
    parallelism: String,
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let opt = Opt::from_args();

    let bounds = parse_pair(&opt.pixels, 'x').expect("error parsing image dimensions");
    let upper_left = parse_complex(&opt.upper_left).expect("error parsing upper left corner point");
    let lower_right =
        parse_complex(&opt.lower_right).expect("error parsing lower right corner point");

    let mut pixels = vec![0; bounds.0 * bounds.1];

    println!(
        "Rendering {}x{} image with corners at ({},{}) and ({},{})",
        bounds.0, bounds.1, upper_left.re, upper_left.im, lower_right.re, lower_right.im
    );

    println!(
        "Using {} threads and '{}' parallelism",
        opt.threads, opt.parallelism
    );

    let elapsed_time = match opt.parallelism.as_str() {
        "single" => do_single(&mut pixels, bounds, upper_left, lower_right),
        "bands" => do_bands(&opt, &mut pixels, bounds, upper_left, lower_right),
        "rayon" => do_rayon(&opt, &mut pixels, bounds, upper_left, lower_right),
        _ => unreachable!("Invalid parallelism mode"),
    };

    write_image(&opt.file, &pixels, bounds).expect("error writing PNG file");

    println!(
        "Done in {} sec {} msec! Picture is written to {}\n",
        elapsed_time.as_secs(),
        elapsed_time.subsec_millis(),
        args[1]
    );
}

fn do_rayon(
    opt: &Opt,
    pixels: &mut Vec<u8>,
    bounds: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) -> std::time::Duration {
    ThreadPoolBuilder::new()
        .num_threads(opt.threads)
        .build_global()
        .unwrap();
    // Scope of slicing up `pixels` into horizontal bands.
    let start_time = Instant::now();
    {
        let bands: Vec<(usize, &mut [u8])> = pixels.chunks_mut(bounds.0).enumerate().collect();

        bands.into_par_iter().for_each(|(i, band)| {
            let top = i;
            let band_bounds = (bounds.0, 1);
            let band_upper_left = pixel_to_point(bounds, (0, top), upper_left, lower_right);
            let band_lower_right =
                pixel_to_point(bounds, (bounds.0, top + 1), upper_left, lower_right);
            render(band, band_bounds, band_upper_left, band_lower_right);
        });
    }
    let elapsed_time = start_time.elapsed();
    elapsed_time
}

fn do_bands(
    opt: &Opt,
    pixels: &mut Vec<u8>,
    bounds: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) -> std::time::Duration {
    let rows_per_band = bounds.1 / opt.threads + 1;

    let start_time = Instant::now();
    {
        let bands: Vec<&mut [u8]> = pixels.chunks_mut(rows_per_band * bounds.0).collect();
        crossbeam::scope(|spawner| {
            for (i, band) in bands.into_iter().enumerate() {
                let top = rows_per_band * i;
                let height = band.len() / bounds.0;
                let band_bounds = (bounds.0, height);
                let band_upper_left = pixel_to_point(bounds, (0, top), upper_left, lower_right);
                let band_lower_right =
                    pixel_to_point(bounds, (bounds.0, top + height), upper_left, lower_right);

                spawner.spawn(move |_| {
                    render(band, band_bounds, band_upper_left, band_lower_right);
                });
            }
        })
        .unwrap();
    }
    let elapsed_time = start_time.elapsed();
    elapsed_time
}

fn do_single(
    pixels: &mut Vec<u8>,
    bounds: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) -> std::time::Duration {
    let start_time = Instant::now();
    {
        render(pixels, bounds, upper_left, lower_right);
    }
    let elapsed_time = start_time.elapsed();
    elapsed_time
}
