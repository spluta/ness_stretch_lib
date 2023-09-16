use crossbeam_utils::thread;
use rand::Rng;
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use std::time::SystemTime;

const MAX_SLICES: usize = 10;

pub struct NessStruct {
    pub max_win_size: usize,
    win_size_divisor: usize,
    pub num_channels: usize,
    pub out_frame_size: usize,
    pub num_slices: usize,
    pub win_lens: Vec<usize>,
    pub hops: Vec<f64>,
    pub loops: usize,
    in_wins: Vec<Vec<f64>>,
    filters: Vec<Vec<Vec<f64>>>,
    filter_on: usize,
    extreme: usize,
    paulstretch_win_size: usize,
    last_frames: Vec<Vec<f64>>,
    pub in_chunk: Vec<Vec<f64>>,
    pub stored_chunk: Vec<Vec<f64>>,
}

impl NessStruct {
    pub fn new(dur_mult: f64, max_win_size: usize, win_size_divisor: usize, num_channels: usize, num_slices: usize, filter_on: usize, mut extreme: usize, paulstretch_win_size: usize, verbosity: usize) -> NessStruct {
        
        //this is the size the frames that are calculated by process_sliced_chunk ==
        //[processed audio(max_win_size)][last_frame1][last_frame2][last_frame3][last_frame4] (for the 4 possible subslices of the slice)
        let out_frame_size: usize = max_win_size * 3;
        
        //256 is always the smallest win_lens, 131072 always the largest (the extras just don't get used)
        let mut win_lens = vec![0_usize; 0];
        let mut hops = vec![0_f64; 0];
        for iter in 0..MAX_SLICES {
            let size = u32::pow(2, 8 + iter as u32);
            //pushes the window sizes into the vector
            win_lens.push(size as usize);
            //pushes the hopsize for each slice into the vector
            hops.push((size as f64 / 2.0) / dur_mult);
        }
        
        
        //creates a vector of fft cutoff bins based on the number of spectral slices
        //the extreme versions can split those cuttoffs into 2 and 4 more subslices
        let cut_max = max_win_size as f64 / 512.0;
        let mut cut_offs = vec![vec![0.0_f64; 0]; MAX_SLICES];
        for iter in 0..MAX_SLICES {
            let cutty: Vec<f64>;
            //add low_cut, then hi_cut
            if iter == (num_slices - 1) {
                cutty = vec![
                1.0,
                cut_max / 4.0,
                cut_max / 2.0,
                3.0 * cut_max / 4.0,
                cut_max,
                ];
            } else {
                cutty = vec![
                cut_max / 2.0,
                5.0 * cut_max / 8.0,
                3.0 * cut_max / 4.0,
                7.0 * cut_max / 8.0,
                cut_max,
                ];
            }
            if num_slices == 1 {
                cut_offs = vec![cutty; MAX_SLICES];
            } else {
                cut_offs[iter] = cutty;
            }
        }

        let mut last_frames = vec![vec![0.0; 0]; 10];
        for iter in 0..10 {
            last_frames[iter] = vec![0.0; win_lens[iter] * 2 * num_channels];
        }

        let in_chunk = vec![vec![0.0; max_win_size*2]; num_channels];
        let stored_chunk = vec![vec![0.0; max_win_size*2]; num_channels];
        
        //this reconfigures the number of ifft loops and arrangement of the cut_offs depending on the extreme algorithm setting
        let mut loops = 1;
        match extreme {
            2 => {
                loops = 4;
            }
            3 => {
                loops = 2;
            }
            _ => {

            }
        }
        for iter in 0..num_slices {

            match extreme {
                0 => {
                    cut_offs[iter][1] = cut_offs[iter][4];
                }
                1 => {
                    cut_offs[iter][1] = cut_offs[iter][4];
                }
                3 => {
                    cut_offs[iter][1] = cut_offs[iter][2];
                    cut_offs[iter][2] = cut_offs[iter][4];
                }
                _ => {
                    cut_offs[iter][1] = cut_offs[iter][4];
                    if extreme < 4 {
                        extreme = 0
                    }
                }
            }
        }

        if verbosity == 1 {
            println!("window sizes {:?}", &win_lens[0..num_slices]);
            println!("spectral cut offs {:?}", cut_offs);
        }

       // let mut filters = vec![vec![0.0_f64; 0]; num_slices];

        let mut filters: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; 0]; 4]; MAX_SLICES];
        let mut in_wins: Vec<Vec<f64>> = vec![vec![0.0; 0]; MAX_SLICES];
        //let mut ness_wins: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; 0]; 101]; num_slices];

        for iter in 0..MAX_SLICES {
            for iter2 in 0..loops {
                let filt_win = make_lr_bp_window(win_lens[iter]/2 + 1, cut_offs[iter][iter2], cut_offs[iter][iter2 + 1], 64.0);
                filters[iter][iter2] = filt_win;
            }
            // for iter2 in 0..=100 {
            //     let ness_win = make_ness_window(win_lens[iter], iter as f64/100.0);
            //     ness_wins[iter][iter2] = ness_win;
            // }

            in_wins[iter] = make_paul_window(win_lens[iter]); //uses a paul window on the input
        }

        NessStruct {max_win_size, win_size_divisor, num_channels, out_frame_size, num_slices, win_lens, hops, loops, in_wins, //ness_wins, 
            filters, filter_on, extreme, paulstretch_win_size,
            //last_frame0,last_frame1,last_frame2,last_frame3,last_frame4,last_frame5,last_frame6,last_frame7,last_frame8,last_frame9,
            last_frames,
            in_chunk, stored_chunk
        }
    }
}


pub fn process_file(file_name: String, dur_mult: f64, extreme: usize, mut num_slices: usize, num_output_blocks: usize, verbosity: usize, filter_on: usize, paulstretch_win_size: usize, out_file: String) {
    //reading the sound file using hound
    //only works with wav files - would be great to replace this with something that works with other formats
    
    let mut sound_file = hound::WavReader::open(file_name).unwrap();
    let mut intemp = vec![0.0; 0];
    
    //loads the sound file into intemp
    //checks to see the format of the sound file and converts all input (float, int16, int24, etc) to f64
    if sound_file.spec().sample_format == hound::SampleFormat::Float {
        intemp.append(
            &mut sound_file
            .samples::<f32>()
            .map(|x| x.unwrap() as f64)
            .collect::<Vec<_>>(),
        );
    } else {
        intemp.append(
            &mut sound_file
            .samples::<i32>()
            .map(|x| x.unwrap() as f64)
            .collect::<Vec<_>>(),
        );
        let bits = sound_file.spec().bits_per_sample;
        for iter in 0..intemp.len() {
            intemp[iter] = intemp[iter] / (f64::powf(2.0, bits as f64));
        }
    };
    
    
    //if the sample rate is 88.2K or above, the largest window will be 131072, otherwise 65536
    //let mut sr_mult = 1;
    let sample_rate = sound_file.spec().sample_rate;
    
    // if sample_rate >= 88200 {
    //     sr_mult = 2;
    // } else if sample_rate >= 176400 {
    //     sr_mult = 4;
    // }
    
    let max_win_size: usize = 65536 * (sample_rate as usize/44100);
    // let mut max_win_size = 16384 * (sample_rate as usize/44100);
    // if num_slices > 1 {
    //     max_win_size = usize::pow(2, 7+num_slices as u32) * (sample_rate as usize/44100);
    // }
    
    if verbosity == 1 {
        println!("Max Window Size: {}", max_win_size);
        if num_slices == 1 {
            println!("PaulStretch window size: {:?}", i32::pow(2,8+(paulstretch_win_size+4) as u32));
        }
    }
    
    //chunks the interleved file into chunks of size channels
    //then transposes the interleaved file into individual vectors for each channel
    let chunked: Vec<Vec<f64>> = intemp
    .chunks(sound_file.spec().channels as usize)
    .map(|x| x.to_vec())
    .collect();
    let mut channels = transpose(chunked);
    
    //then creates output vectors for each channel as well
    //let out_channels: Vec<Vec<f32>> = vec![vec![0.0_f32; 0]; sound_file.spec().channels as usize];
    
    let now = SystemTime::now();
    
    //the higher sample rates can have 10 slices
    if sample_rate < 88200 && num_slices > 9 {
        num_slices = MAX_SLICES - 1;
    } else if sample_rate >= 88200 && num_slices > 9 {
        num_slices = MAX_SLICES;
    }
    if verbosity == 1 {
        println!("The audio file will be sliced into {} slices", num_slices);
    }
    
    let num_channels = channels.len();
    
    let mut ness_struct: NessStruct = NessStruct::new(dur_mult, max_win_size, 1, num_channels, num_slices, filter_on, extreme, paulstretch_win_size, verbosity);
    
    let mut indata = vec![vec![0.0_f64; 0]; num_channels]; //empty array
    let in_size = channels[0].len(); //max_win_size + 
    
    //let frames_to_add = 2 * max_win_size - (in_size % max_win_size);
    //println!("framestoadd {}", frames_to_add);
    for i in 0..num_channels {
        indata[i].append(&mut channels[i]); //adds the channel data to indata
        //indata[i].append(&mut vec![0.0_f64; frames_to_add]); //adds 0s at the end of indata so that the vector size is divisible by max_win_size
        // println!("indatasize {}", indata[i].len());
    }
    

    let mut num_chunks = (in_size as f64 / max_win_size as f64 * dur_mult) as usize;
    if num_output_blocks > 0 {
        num_chunks = num_output_blocks;
    }

    let mut chunk_points = vec![0; num_chunks];
    
    for iter in 0..num_chunks {
        chunk_points[iter] = ((iter * max_win_size) as f64 / dur_mult) as usize;
    }
    
    //hound is the wav reader and writer
    let spec = hound::WavSpec {
        channels: num_channels as u16,
        sample_rate: sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    
    let mut writer = hound::WavWriter::create(out_file, spec).unwrap();
    
    let start_iter = 0;
    
    //go through the chunk_points, making max_win_size chunks of audio
    for iter in 0..num_chunks {
        //only process the chunk if iter>
        if iter >= start_iter && (iter < (num_chunks - start_iter)) {

            if iter % 25 == 0 && verbosity == 1 {
                println!("chunk {} of {}", iter, num_chunks)
            }

            for chan in 0..num_channels {
                for sample in 0..(max_win_size*2){
                    let point = chunk_points[iter]+sample;
                    if point < indata[0].len() {
                        ness_struct.in_chunk[chan][sample] = indata[chan][point];
                    } else {
                        ness_struct.in_chunk[chan][sample] = 0.0f64;
                    }
                    
                }
            }
            
            ness_struct.stored_chunk = process_chunk(&mut ness_struct);
            for samp in 0..(max_win_size) {
                (0..channels.len()).for_each(|chan| {
                    writer.write_sample(ness_struct.stored_chunk[chan][samp] as f32).unwrap();
                });
            }
        }
        
    }
    
    //close the output file
    writer.finalize().unwrap();
    if verbosity == 1 {
        println!("{:?}", now.elapsed())
    };
    
}


pub fn process_chunk(
    ness_struct: &mut NessStruct,
    
) -> Vec<Vec<f64>> {
    
    //grab all the info from the ness_struct
    let win_lens = &ness_struct.win_lens;
    let out_frame_size = ness_struct.max_win_size*3;
    let num_channels = ness_struct.num_channels;
    let hops = &ness_struct.hops;
    let loops = &ness_struct.loops;
    let in_wins = &ness_struct.in_wins;
    let filters = &ness_struct.filters;
    let extreme = ness_struct.extreme;
    let max_win_size = ness_struct.max_win_size; //use the max_win_size by default, but be able to set it
    let win_size_divisor = ness_struct.win_size_divisor;
    let num_slices = ness_struct.num_slices;
    let filter_on: usize = ness_struct.filter_on;
    let indata = &ness_struct.in_chunk;
    let chunk_point = 0;
    
    let mut out_temp0 = vec![0.0; out_frame_size];
    let mut out_temp1 = vec![0.0; out_frame_size];
    let mut out_temp2 = vec![0.0; out_frame_size];
    let mut out_temp3 = vec![0.0; out_frame_size];
    let mut out_temp4 = vec![0.0; out_frame_size];
    let mut out_temp5 = vec![0.0; out_frame_size];
    let mut out_temp6 = vec![0.0; out_frame_size];
    let mut out_temp7 = vec![0.0; out_frame_size];
    let mut out_temp8 = vec![0.0; out_frame_size];
    let mut out_temp9 = vec![0.0; out_frame_size];
    
    let mut out_frame0 = vec![0.0; out_frame_size * num_channels];
    let mut out_frame1 = vec![0.0; out_frame_size * num_channels];
    let mut out_frame2 = vec![0.0; out_frame_size * num_channels];
    let mut out_frame3 = vec![0.0; out_frame_size * num_channels];
    let mut out_frame4 = vec![0.0; out_frame_size * num_channels];
    let mut out_frame5 = vec![0.0; out_frame_size * num_channels];
    let mut out_frame6 = vec![0.0; out_frame_size * num_channels];
    let mut out_frame7 = vec![0.0; out_frame_size * num_channels];
    let mut out_frame8 = vec![0.0; out_frame_size * num_channels];
    let mut out_frame9 = vec![0.0; out_frame_size * num_channels];
    
    let mut last_frame0 = ness_struct.last_frames[0].clone();
    let mut last_frame1 = ness_struct.last_frames[1].clone();
    let mut last_frame2 = ness_struct.last_frames[2].clone();
    let mut last_frame3 = ness_struct.last_frames[3].clone();
    let mut last_frame4 = ness_struct.last_frames[4].clone();
    let mut last_frame5 = ness_struct.last_frames[5].clone();
    let mut last_frame6 = ness_struct.last_frames[6].clone();
    let mut last_frame7 = ness_struct.last_frames[7].clone();
    let mut last_frame8 = ness_struct.last_frames[8].clone();
    let mut last_frame9 = ness_struct.last_frames[9].clone();

    //println!("{}", num_slices);
    
    if num_slices == 1 {
        let mut paul_win_index:usize = 5;
        if ness_struct.paulstretch_win_size == 2 {paul_win_index = 6};
        if ness_struct.paulstretch_win_size == 3 {paul_win_index = 7};

        last_frame0 = ness_struct.last_frames[paul_win_index].clone();

        for chan_num in 0..num_channels {
            
            let win_len = win_lens[paul_win_index];
            
            //out_temp will be the chunk of audio to write, then 4 "last_frames", one for each of the possible subslices
            out_temp0 = process_sliced_chunk(
                &indata[chan_num],
                chunk_point, 
                win_len,
                0,//filter_on,  //force the filter to be off
                &hops[paul_win_index],
                *loops,
                in_wins[paul_win_index].clone(),
                //ness_wins[5].clone(),
                filters[paul_win_index].clone(),
                last_frame0.clone(),
                chan_num,
                extreme,
                max_win_size,
                out_frame_size,
                win_size_divisor
            );
            
            //put the last_frame data back into the last frame so it is there when we loop around to the next chunk
            for i in 0..(win_len * 2) {
                last_frame0[chan_num * win_len * 2 + i] =
                out_temp0[max_win_size + i];
            }
            //grab the out_frame from the out_temp
            //the out_frame is a flat array with spaces for all channels of output audio
            //it is stored [channel0][channel1]..etc, but is flat
            for i in 0..max_win_size {
                out_frame0[chan_num * max_win_size + i] = out_temp0[i];
            }
            
        };

        ness_struct.last_frames[paul_win_index] = last_frame0.clone();
        
        
    } else {
        
        //super ugly, but as far as I know, this is the only way to borrow from last_frame and then write back into it
        thread::scope(|s| {
            s.spawn(|_| {
                for chan_num in 0..num_channels {
                    //out_temp will be the chunk of audio to write, then 4 "last_frames", one for each of the possible subslices
                    out_temp0 = process_sliced_chunk(
                        &indata[chan_num],
                        chunk_point,
                        win_lens[0],
                        filter_on,
                        &hops[0],
                        *loops,
                        in_wins[0].clone(),
                        filters[0].clone(),
                        last_frame0.clone(),
                        chan_num,
                        extreme,
                        max_win_size,
                        out_frame_size,
                        win_size_divisor
                    );
                    
                    //put the last_frame data back into the last frame so it is there when we loop around to the next chunk
                    for i in 0..(win_lens[0] * 2) {
                        last_frame0[chan_num * win_lens[0] * 2 + i] =
                        out_temp0[max_win_size + i];
                    }
                    //grab the out_frame from the out_temp
                    //the out_frame is a flat array with spaces for all channels of output audio
                    //it is stored [channel0][channel1]..etc, but is flat
                    for i in 0..max_win_size {
                        out_frame0[chan_num * max_win_size + i] = out_temp0[i];
                    }
                }
            });
            if num_slices > 1 {
                s.spawn(|_| {
                    for chan_num in 0..num_channels {
                        out_temp1 = process_sliced_chunk(
                            &indata[chan_num],
                            chunk_point,
                            win_lens[1],
                            filter_on,
                            &hops[1],
                            *loops,
                            in_wins[1].clone(),
                            filters[1].clone(),
                            last_frame1.clone(),
                            chan_num,
                            extreme,
                            max_win_size,
                            out_frame_size,
                            win_size_divisor
                        );
                        for i in 0..(win_lens[1] * 2) {
                            last_frame1[chan_num * win_lens[1] * 2 + i] =
                            out_temp1[max_win_size + i];
                        }
                        for i in 0..max_win_size {
                            out_frame1[chan_num * max_win_size + i] = out_temp1[i];
                        }
                    }
                });
            };
            if num_slices > 2 {
                s.spawn(|_| {
                    for chan_num in 0..num_channels {
                        out_temp2 = process_sliced_chunk(
                            &indata[chan_num],
                            chunk_point,
                            win_lens[2],
                            filter_on,
                            &hops[2],
                            *loops,
                            in_wins[2].clone(),
                            filters[2].clone(),
                            last_frame2.clone(),
                            chan_num,
                            extreme,
                            max_win_size,
                            out_frame_size,
                            win_size_divisor
                        );
                        for i in 0..(win_lens[2] * 2) {
                            last_frame2[chan_num * win_lens[2] * 2 + i] =
                            out_temp2[max_win_size + i];
                        }
                        for i in 0..max_win_size {
                            out_frame2[chan_num * max_win_size + i] = out_temp2[i];
                        }
                    }
                });
            };
            if num_slices > 3 {
                s.spawn(|_| {
                    for chan_num in 0..num_channels {
                        out_temp3 = process_sliced_chunk(
                            &indata[chan_num],
                            chunk_point,
                            win_lens[3],
                            filter_on,
                            &hops[3],
                            *loops,
                            in_wins[3].clone(),
                            filters[3].clone(),
                            last_frame3.clone(),
                            chan_num,
                            extreme,
                            max_win_size,
                            out_frame_size,
                            win_size_divisor
                        );
                        for i in 0..(win_lens[3] * 2) {
                            last_frame3[chan_num * win_lens[3] * 2 + i] =
                            out_temp3[max_win_size + i];
                        }
                        for i in 0..max_win_size {
                            out_frame3[chan_num * max_win_size + i] = out_temp3[i];
                        }
                    }
                });
            };
            if num_slices > 4 {
                s.spawn(|_| {
                    for chan_num in 0..num_channels {
                        out_temp4 = process_sliced_chunk(
                            &indata[chan_num],
                            chunk_point,
                            win_lens[4],
                            filter_on,
                            &hops[4],
                            *loops,
                            in_wins[4].clone(),
                            filters[4].clone(),
                            last_frame4.clone(),
                            chan_num,
                            extreme,
                            max_win_size,
                            out_frame_size,
                            win_size_divisor
                        );
                        for i in 0..(win_lens[4] * 2) {
                            last_frame4[chan_num * win_lens[4] * 2 + i] =
                            out_temp4[max_win_size + i];
                        }
                        for i in 0..max_win_size {
                            out_frame4[chan_num * max_win_size + i] = out_temp4[i];
                        }
                    }
                });
            };
            if num_slices > 5 {
                s.spawn(|_| {
                    for chan_num in 0..num_channels {
                        out_temp5 = process_sliced_chunk(
                            &indata[chan_num],
                            chunk_point,
                            win_lens[5],
                            filter_on,
                            &hops[5],
                            *loops,
                            in_wins[5].clone(),
                            filters[5].clone(),
                            last_frame5.clone(),
                            chan_num,
                            extreme,
                            max_win_size,
                            out_frame_size,
                            win_size_divisor
                        );
                        for i in 0..(win_lens[5] * 2) {
                            last_frame5[chan_num * win_lens[5] * 2 + i] =
                            out_temp5[max_win_size + i];
                        }
                        for i in 0..max_win_size {
                            out_frame5[chan_num * max_win_size + i] = out_temp5[i];
                        }
                    }
                });
            };
            if num_slices > 6 {
                s.spawn(|_| {
                    for chan_num in 0..num_channels {
                        out_temp6 = process_sliced_chunk(
                            &indata[chan_num],
                            chunk_point,
                            win_lens[6],
                            filter_on,
                            &hops[6],
                            *loops,
                            in_wins[6].clone(),
                            filters[6].clone(),
                            last_frame6.clone(),
                            chan_num,
                            extreme,
                            max_win_size,
                            out_frame_size,
                            win_size_divisor
                        );
                        for i in 0..(win_lens[6] * 2) {
                            last_frame6[chan_num * win_lens[6] * 2 + i] =
                            out_temp6[max_win_size + i];
                        }
                        for i in 0..max_win_size {
                            out_frame6[chan_num * max_win_size + i] = out_temp6[i];
                        }
                    }
                });
            };
            if num_slices > 7 {
                s.spawn(|_| {
                    for chan_num in 0..num_channels {
                        out_temp7 = process_sliced_chunk(
                            &indata[chan_num],
                            chunk_point,
                            win_lens[7],
                            filter_on,
                            &hops[7],
                            *loops,
                            in_wins[7].clone(),
                            filters[7].clone(),
                            last_frame7.clone(),
                            chan_num,
                            extreme,
                            max_win_size,
                            out_frame_size,
                            win_size_divisor
                        );
                        for i in 0..(win_lens[7] * 2) {
                            last_frame7[chan_num * win_lens[7] * 2 + i] =
                            out_temp7[max_win_size + i];
                        }
                        for i in 0..max_win_size {
                            out_frame7[chan_num * max_win_size + i] = out_temp7[i];
                        }
                    }
                });
            };
            if num_slices > 8 {
                s.spawn(|_| {
                    for chan_num in 0..num_channels {
                        out_temp8 = process_sliced_chunk(
                            &indata[chan_num],
                            chunk_point,
                            win_lens[8],
                            filter_on,
                            &hops[8],
                            *loops,
                            in_wins[8].clone(),
                            filters[8].clone(),
                            last_frame8.clone(),
                            chan_num,
                            extreme,
                            max_win_size,
                            out_frame_size,
                            win_size_divisor
                        );
                        for i in 0..(win_lens[8] * 2) {
                            last_frame8[chan_num * win_lens[8] * 2 + i] =
                            out_temp8[max_win_size + i];
                        }
                        for i in 0..max_win_size {
                            out_frame8[chan_num * max_win_size + i] = out_temp8[i];
                        }
                    }
                });
            };
            if num_slices > 9 {
                s.spawn(|_| {
                    for chan_num in 0..num_channels {
                        out_temp9 = process_sliced_chunk(
                            &indata[chan_num],
                            chunk_point,
                            win_lens[9],
                            filter_on,
                            &hops[9],
                            *loops,
                            in_wins[9].clone(),
                            filters[9].clone(),
                            last_frame9.clone(),
                            chan_num,
                            extreme,
                            max_win_size,
                            out_frame_size,
                            win_size_divisor
                        );
                        for i in 0..(win_lens[9] * 2) {
                            last_frame9[chan_num * win_lens[9] * 2 + i] =
                            out_temp9[max_win_size + i];
                        }
                        for i in 0..max_win_size {
                            out_frame9[chan_num * max_win_size + i] = out_temp9[i];
                        }
                    }
                });
            };
        })
        .unwrap();

        //copy the last_frame data back into the ness_struct
        ness_struct.last_frames[0] = last_frame0.clone();
        ness_struct.last_frames[1] = last_frame1.clone();
        ness_struct.last_frames[2] = last_frame2.clone();
        ness_struct.last_frames[3] = last_frame3.clone();
        ness_struct.last_frames[4] = last_frame4.clone();
        ness_struct.last_frames[5] = last_frame5.clone();
        ness_struct.last_frames[6] = last_frame6.clone();
        ness_struct.last_frames[7] = last_frame7.clone();
        ness_struct.last_frames[8] = last_frame8.clone();
        ness_struct.last_frames[9] = last_frame9.clone();
    }

    
    
    let mut out_data: Vec<Vec<f64>> = vec![vec![0.0; max_win_size]; num_channels];
    
    //out_frameN has the M channels of audio spread out accross a single array
    //with max_win_size frames per channel
    for chan_num in 0..num_channels {
        let read_point = chan_num * max_win_size;
        for i in 0..max_win_size {
            out_data[chan_num][i] = out_frame0[read_point + i]
            + out_frame1[read_point + i]
            + out_frame2[read_point + i]
            + out_frame3[read_point + i]
            + out_frame4[read_point + i]
            + out_frame5[read_point + i]
            + out_frame6[read_point + i]
            + out_frame7[read_point + i]
            + out_frame8[read_point + i]
            + out_frame9[read_point + i];
        }
    }
    //out_data is a multidimensional array with max_win_size frames per channel
    return out_data;
}

//this is the code that does the actual randomizing of phases
fn process_microframe(
    spectrum: Vec<Complex<f64>>,
    last_frame: &[f64],
    filt_win: Vec<f64>,
    filter_on: usize,
    extreme: usize,
) -> Vec<f64> {
    let half_win_len = spectrum.len() - 1;
    let win_len = half_win_len * 2;
    
    //sets up the ifft planner
    let mut real_planner = RealFftPlanner::<f64>::new();
    let ifft = real_planner.plan_fft_inverse(win_len);
    let mut out_frame = ifft.make_output_vec();
    let mut fin_out_frame = ifft.make_output_vec();
    let mut flipped_frame = vec![0.0; win_len];
    let mut spectrum_out = real_planner.plan_fft_forward(win_len).make_output_vec();
    
    //the correlation values used
    let mut correlation = 0.0;
    let mut corr_temp = 0.0;
    let mut corr_abs;
    let mut c_a_temp = 0.0;
    let mut num_ffts = 1;
    
    //sets up the ffts based on the extreme setting
    if extreme == 1 {
        num_ffts = 10
    }
    if extreme == 3 {
        num_ffts = 3
    }
    if extreme > 3 {
        num_ffts = extreme
    }
    
    //goes through and makes all the ffts to compare correlation on
    for _count in 0..num_ffts {
        //0s the bins and randomizes the phases
        for iter in 1..spectrum.len()-1 {
            
            let mut temp = spectrum[iter].to_polar();
            if filter_on == 1 {temp.0 = temp.0 * filt_win[iter]}; //multiply by the filter if filter is on
            temp.1 = rand::thread_rng().gen_range(-PI/2.0..PI/2.0);
            spectrum_out[iter] = Complex::from_polar(temp.0, temp.1);
        }
        
        assert_eq!(spectrum_out.len(), spectrum.len());
        //performs the ifft
        ifft.process(&mut spectrum_out, &mut out_frame).unwrap();
        
        //gets half the frame and checks correlation with the previous frame
        //let half_vec0 = &last_frame[win_len/2..];
        let half_current_frame = &out_frame[..half_win_len];
        
        let temp_sum: f64 = last_frame.iter().sum();
        if temp_sum != 0.0 {
            let r: f64 = last_frame
            .iter()
            .zip(half_current_frame.iter())
            .map(|(x, y)| x * y)
            .sum();
            let s: f64 = last_frame
            .iter()
            .zip(last_frame.iter())
            .map(|(x, y)| x * y)
            .sum();
            corr_temp = r / s;
        }
        corr_abs = corr_temp.abs();
        
        //if the correlation is better use this one
        if corr_abs > c_a_temp {
            correlation = corr_temp;
            c_a_temp = correlation.abs();
            fin_out_frame = out_frame.clone();
        }
    }
    corr_abs = correlation.abs();
    if correlation == 0.0 {
        fin_out_frame = out_frame.clone()
    }
    //inverts the randomized signal if the correlation is negative
    for i in 0..win_len {
        if correlation < 0.0 {
            flipped_frame[i] = -1.0 * fin_out_frame[i];
        } else {
            flipped_frame[i] = fin_out_frame[i];
        }
    }

    //gets the ness_window
    if corr_abs>1.0 {corr_abs=1.0};

    let ness_window = make_ness_window(win_len, corr_abs);
    
    let mut out_frame2 = vec![0.0; win_len];
    
    //multiples the start of the ness_window by the start of the frame
    //and the end of the ness_window by the end of the frame
    for i in 0..half_win_len {
        out_frame2[i] =
        flipped_frame[i] * ness_window[i] + (last_frame[i] * ness_window[half_win_len-1-i]);
    }
    //add the second half of the flipped frame (no ness_window) to check for correlation on the next loop
    for i in 0..half_win_len {
        out_frame2[i + half_win_len] = flipped_frame[i + half_win_len];
    }
    
    //returns a frame that contains the half win_len frame multiplied by the ness_window followed by the flipped frame (necessary for checking the next correlation)
    
    return out_frame2;
}



//creates a chunk of audio that is the size of the max_win_size
fn process_sliced_chunk(
    indata: &Vec<f64>,
    chunk_point: usize,
    win_len: usize,
    filter_on: usize,
    hop: &f64,
    loops: usize,
    in_win: Vec<f64>,
    //ness_wins: Vec<Vec<f64>>,
    filters: Vec<Vec<f64>>,
    mut last_frame: Vec<f64>,
    chan_num: usize,
    //num_slices: usize,
    extreme: usize,
    max_win_size: usize,
    out_frame_size: usize,
    win_size_divisor: usize,
) -> Vec<f64> {

    let half_win_len = win_len / 2;
    
    
    //the vector of stretch points contains the points where we will be reading from the indata
    let mut stretch_points = vec![0; (max_win_size / half_win_len) as usize];
    for iter in 0..stretch_points.len() {
        stretch_points[iter] =
        chunk_point + (hop * iter as f64) as usize + (max_win_size / 2 - half_win_len) as usize;
    }
    
    //the vector of out_points contains the points where we will be writing into the out_chunk buffer
    let mut out_points = vec![0; stretch_points.len()];
    for iter in 0..out_points.len() {
        out_points[iter] = iter * half_win_len;
    }
    
    //this is the audio we will be writing to disk plus the 4 extreme 2 last_frame vectors
    //it will store the full audio chunk
    let mut out_chunk = vec![0.0; out_frame_size];
    
    //this is the lookup location into the last_frame - since the last_frame contains "num_channels" locations with 4 half_win sized frames at each location
    let chan_point = chan_num * win_len * 2;
    
    //big loop over the stretch points
    for big_iter in 0..(stretch_points.len()/win_size_divisor) {
        //for efficiency, does the fft once for the frame
        let mut real_planner = RealFftPlanner::<f64>::new();
        let fft = real_planner.plan_fft_forward(win_len);
        let mut spectrum = fft.make_output_vec();
        let mut part = vec![0.0; win_len];
        for i in 0..win_len {
            part[i] = indata[stretch_points[big_iter] + i] * in_win[i];
        }
        fft.process(&mut part, &mut spectrum).unwrap();
        
        //will loop once, twice, or 4 times depending on algorithm
        for i in 0..loops {
            //makes the linquitz-riley window at the cuttoff points
            let filt_win = filters[i].clone();
            let last_frame_slice =
            &last_frame[chan_point + i * half_win_len..chan_point + (i + 1) * half_win_len];
            
            //process_microframe does the actual processing of the phase and returns the phase randomized frame
            let out_frame =
            process_microframe(spectrum.clone(), last_frame_slice, filt_win, filter_on, extreme);//&ness_wins,
            
            //get the current frame to return as the last
            for i2 in 0..half_win_len {
                last_frame[chan_point + i * half_win_len + i2] = out_frame[half_win_len + i2];
            }
            //put the half frame sound output into the out_data starting at the outpoints
            let out_spot = out_points[big_iter] as usize;
            for i2 in 0..half_win_len {
                out_chunk[out_spot + i2] += out_frame[i2] / win_len as f64;
            }
        }
    }
    
    //put the last frame output into the out_chunk at a point based on the channel being processed
    for i in 0..win_len * 2 {
        out_chunk[max_win_size + i] = last_frame[chan_point + i];
    }
    return out_chunk;
}

//makes the the first half of the ness window in accordance with the correlation number provided
fn make_ness_window(len: usize, correlation: f64) -> Vec<f64> { 
    let lendiv2 = len/2;
    let mut floats: Vec<f64> = vec![0.0; lendiv2];
    let mut vals: Vec<f64> = vec![0.0; lendiv2];
    //lendiv2 = lendiv2 ;
    for iter in 0..(lendiv2) {
        floats[iter] = iter as f64 / ((len-1) as f64 / 2.0);
    }
    //floats.push(0.0);
    for iter in 0..lendiv2 {
        let fs = f64::powf((floats[iter] * PI / 2.0).tan(), 2.0);
        vals[iter] = fs * (1.0 / (1.0 + (2.0 * fs * (correlation)) + f64::powf(fs, 2.0))).sqrt();
    }
    return vals;
}

//makes the linkwitz-riley fft crossfade window, which effectively 0s out the bins wanted in the spectral slice
//high pass, low pass, and bandbass versions
fn make_lr_lp_window(len: usize, hi_bin: f64, order: f64) -> Vec<f64> {
    let mut filter = vec![1.0; len];
    if hi_bin != 0.0 {
        for i in 0..len {
            filter[i] = 1.0 / (1.0 + (f64::powf(i as f64 / hi_bin, order)));
        }
    }
    return filter;
}

fn make_lr_hp_window(len: usize, low_bin: f64, order: f64) -> Vec<f64> {
    let mut filter = vec![1.0; len];
    if low_bin != 0.0 {
        for i in 0..len {
            filter[i] = 1.0 - (1.0 / (1.0 + (f64::powf(i as f64 / low_bin, order))));
        }
    }
    return filter;
}

fn make_lr_bp_window(len: usize, low_bin: f64, hi_bin: f64, order: f64) -> Vec<f64> {
    let filter: Vec<f64>;
    if low_bin <= 0.0 {
        filter = make_lr_lp_window(len, hi_bin, order);
    } else {
        if hi_bin >= (len - 2) as f64 {
            filter = make_lr_hp_window(len, low_bin, order);
        } else {
            let lp = make_lr_lp_window(len, hi_bin, order);
            let hp = make_lr_hp_window(len, low_bin, order);
            filter = lp.iter().zip(hp.iter()).map(|(x, y)| x * y).collect();
        }
    }
    return filter;
}

//the paul stretch window is used on input - might a well be a sine or hann window
fn make_paul_window(len: usize) -> Vec<f64> {
    let mut part = vec![0.0; len];
    for i in 0..len {
        let value = i as f64 / (len as f64 - 1.0) * 2.0 - 1.0;
        let value = f64::powf(1.0 - (f64::powf(value, 2.0)), 1.25);
        part[i] = value;
    }
    return part;
}

//flops a channel array from interleaved clusters to separate files
fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>>
where
T: Clone,
{
    assert!(!v.is_empty());
    (0..v[0].len())
    .map(|i| v.iter().map(|inner| inner[i].clone()).collect::<Vec<T>>())
    .collect()
}
