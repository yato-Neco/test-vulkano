use core::slice;
use std::ffi::c_void;
use std::sync::Arc;
use opencv::highgui::{self, imshow, WINDOW_NORMAL};
use opencv::traits::Boxed;
use vulkano::image::Image;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::VulkanLibrary;

use image::io::Reader as ImageReader;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags},
    instance::InstanceCreateFlags,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    sync::{self, GpuFuture},
};

use opencv::core::{Mat, MatTraitConst, MatTraitConstManual, Mat_AUTO_STEP, Size, CV_32F, CV_32S, CV_32SC3, CV_8UC3};
use opencv::prelude::VideoCaptureTrait;
use opencv::videoio::{CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH};
use opencv::{imgproc, videoio};

fn main() {
    //let image = ImageReader::open("src/cri.png").unwrap().decode().unwrap();
    //let img = image.to_rgb32f().to_vec();
    let img =vec![1920*1080*3];


    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .expect("failed to create instance");

    let physical_device = instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .next()
        .expect("no devices available");

    for family in physical_device.queue_family_properties() {
        println!(
            "Found a queue family with {:?} queue(s)",
            family.queue_count
        );
    }

    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_queue_family_index, queue_family_properties)| {
            queue_family_properties
                .queue_flags
                .contains(QueueFlags::GRAPHICS)
        })
        .expect("couldn't find a graphical queue family") as u32;

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            // here we pass the desired queue family to use by index
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .expect("failed to create device");

    let queue = queues.next().unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));



    let shader = cs::load(device.clone()).expect("failed to create shader module");

    let cs = shader.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(cs);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .expect("failed to create compute pipeline");

    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(device.clone(), Default::default());
    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();

    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .unwrap();
  

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

   

    let work_group_counts = [1920 * 1080, 1, 1];
    //let work_group_counts = [1024, 1, 1];

  


    
    let mut cam = videoio::VideoCapture::new(1, videoio::CAP_ANY).unwrap(); // 0 is the default camera
    cam.set(CAP_PROP_FRAME_WIDTH, 1920 as f64).unwrap();
    cam.set(CAP_PROP_FRAME_HEIGHT, 1080 as f64).unwrap();

    let mut frame =
        unsafe { Mat::new_size(Size::from((1080 as i32, 1920 as i32)), CV_8UC3).unwrap() };

    let mut rgb_frame = unsafe {
            Mat::new_size(Size::from((1080 as i32, 1920 as i32)), CV_8UC3).unwrap()
        };

    

    
     loop {
        cam.read(&mut frame).unwrap();

        //opencv::imgproc::cvt_color(&frame, &mut rgb_frame, imgproc::COLOR_BGR2RGB, 0).unwrap();
        let image= frame.data_bytes().unwrap();

        let image:Vec<u32> = image.iter().map(|p| *p as u32).collect();

        //println!("{}",image[100]);

        let data_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            image,
        )
        .expect("failed to create buffer");

        let descriptor_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            descriptor_set_layout.clone(),
            [WriteDescriptorSet::buffer(0, data_buffer.clone())], // 0 is the binding
            [],
        )
        .unwrap();

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        command_buffer_builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            descriptor_set_layout_index as u32,
            descriptor_set,
        )
        .unwrap()
        .dispatch(work_group_counts)
        .unwrap();
    
        let command_buffer = command_buffer_builder.build().unwrap();

        let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
        future.wait(None).unwrap();  // None is an optional timeout

        //let mut buf = data_buffer.write().unwrap();

        //buf.copy_from_slice(image.as_slice());
        //future.wait(None).unwrap();  // None is an optional timeout

        let image = data_buffer.read().unwrap();

        let image:Vec<u8> = image.iter().map(|p| *p as u8).collect();

        let src1 = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                1080.try_into().unwrap(),
                1920.try_into().unwrap(),
                CV_8UC3,
                image.to_vec().as_mut_ptr() as *mut c_void,
                Mat_AUTO_STEP,
            )
            .unwrap()
        };
       
        imshow("32F Image", &src1).unwrap();

        let key = highgui::wait_key(1).unwrap();
        if key > 0 && key != 255 {
            break;
        }
       
        //future.wait(None).unwrap();
        //println!("更新");
    }
    
    
   
    
    

    /*
    future.wait(None).unwrap();

    let content = data_buffer.read().unwrap();
    //println!("{:?}",content);
    println!("{}", content[0]);
    let content: Vec<u8> = content.iter().map(|p| (*p ) as u8).collect();
    println!("{}", content[0]);

    image::save_buffer(
        "image.png",
        &content,
        image.width(),
        image.height(),
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    for (n, val) in content.iter().enumerate() {
        //assert_eq!(*val, n as u32 * 3);
    }
    */
    


    

    println!("Everything succeeded!");
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path:"src/gray.comp"
    }
}
