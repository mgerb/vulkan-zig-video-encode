const vk = @import("vulkan");
const std = @import("std");
const print = std.debug.print;
const assert = std.debug.assert;
const Device = @import("./vulkan_app.zig").Device;
const Instance = @import("./vulkan_app.zig").Instance;
const VulkanApp = @import("./vulkan_app.zig").VulkanApp;
const h264_parameters = @import("./h264_parameters.zig");

const Display = u32;
const REFERENCE_IMAGE_COUNT = 2;

pub const VideoEncoder = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    vulkan_app: *VulkanApp,
    physical_device: vk.PhysicalDevice,
    device: Device,
    instance: Instance,
    compute_queue_family: u32,
    compute_queue: vk.Queue,
    compute_command_pool: vk.CommandPool,
    encode_queue_family: u32,
    encode_queue: vk.Queue,
    input_images: std.ArrayList(vk.Image),
    input_image_views: std.ArrayList(vk.ImageView),
    height: u32,
    width: u32,
    fps: u32,

    vui: ?vk.StdVideoH264SequenceParameterSetVui = null,
    sps: ?vk.StdVideoH264SequenceParameterSet = null,
    pps: ?vk.StdVideoH264PictureParameterSet = null,

    encode_command_pool: ?vk.CommandPool = null,
    video_session: ?vk.VideoSessionKHR = null,
    video_profile: ?vk.VideoProfileInfoKHR = null,
    video_profile_list: ?vk.VideoProfileListInfoKHR = null,
    video_session_parameters: ?vk.VideoSessionParametersKHR = null,
    encode_session_bind_memory: std.ArrayList(vk.BindVideoSessionMemoryInfoKHR),

    // bit stream
    bit_stream_header: std.ArrayList(u8),
    bit_stream_header_pending: bool = false,
    bit_stream_data: ?[*]const u8 = null,
    bit_stream_memory: ?vk.DeviceMemory = null,
    bit_stream_buffer: ?vk.Buffer = null,

    // images
    chosen_dpb_image_format: ?vk.Format = null,
    dpb_images: std.ArrayList(vk.Image),
    dpb_image_memory: std.ArrayList(vk.DeviceMemory),
    dpb_image_views: std.ArrayList(vk.ImageView),

    chosen_src_image_format: ?vk.Format = null,
    ycbcr_image: ?vk.Image = null,
    ycbcr_image_view: ?vk.ImageView = null,
    ycbcr_image_plane_views: std.ArrayList(vk.ImageView),
    ycbcr_image_memory: ?vk.DeviceMemory = null,

    query_pool: ?vk.QueryPool = null,

    compute_descriptor_set_layout: ?vk.DescriptorSetLayout = null,
    compute_pipeline_layout: ?vk.PipelineLayout = null,
    compute_pipeline: ?vk.Pipeline = null,
    compute_descriptor_sets: std.ArrayList(vk.DescriptorSet),
    descriptor_pool: ?vk.DescriptorPool = null,

    inter_queue_semaphore1: ?vk.Semaphore = null,
    inter_queue_semaphore2: ?vk.Semaphore = null,
    inter_queue_semaphore3: ?vk.Semaphore = null,
    encode_finished_fence: ?vk.Fence = null,

    encode_rate_control_layer_info: ?vk.VideoEncodeRateControlLayerInfoKHR = null,
    encode_h264_rate_control_layer_info: ?vk.VideoEncodeH264RateControlLayerInfoKHR = null,
    encode_rate_control_info: ?vk.VideoEncodeRateControlInfoKHR = null,
    encode_h264_rate_control_info: ?vk.VideoEncodeH264RateControlInfoKHR = null,
    chosen_rate_control_mode: ?vk.VideoEncodeRateControlModeFlagsKHR = null,

    encode_command_buffer: ?vk.CommandBuffer = null,
    compute_command_buffer: ?vk.CommandBuffer = null,

    frame_count: u32 = 0,
    running: bool = false,

    pub fn init(
        allocator: std.mem.Allocator,
        vulkan_app: *VulkanApp,
        physical_device: vk.PhysicalDevice,
        device: Device,
        instance: Instance,
        compute_queue_family: u32,
        compute_queue: vk.Queue,
        compute_command_pool: vk.CommandPool,
        encode_queue_family: u32,
        encode_queue: vk.Queue,
        input_images: std.ArrayList(vk.Image),
        input_image_views: std.ArrayList(vk.ImageView),
        width: u32,
        height: u32,
        fps: u32,
    ) !*Self {
        var self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .vulkan_app = vulkan_app,
            .height = height & ~@as(u32, 1),
            .width = width & ~@as(u32, 1),
            .physical_device = physical_device,
            .device = device,
            .instance = instance,
            .compute_command_pool = compute_command_pool,
            .compute_queue = compute_queue,
            .compute_queue_family = compute_queue_family,
            .encode_queue_family = encode_queue_family,
            .encode_queue = encode_queue,
            .input_images = input_images,
            .input_image_views = input_image_views,
            .fps = fps,

            .bit_stream_header = std.ArrayList(u8).init(allocator),
            .encode_session_bind_memory = std.ArrayList(vk.BindVideoSessionMemoryInfoKHR).init(allocator),

            .dpb_images = std.ArrayList(vk.Image).init(allocator),
            .dpb_image_memory = std.ArrayList(vk.DeviceMemory).init(allocator),
            .dpb_image_views = std.ArrayList(vk.ImageView).init(allocator),
            .ycbcr_image_plane_views = std.ArrayList(vk.ImageView).init(allocator),
            .compute_descriptor_sets = std.ArrayList(vk.DescriptorSet).init(allocator),
        };

        try self.createEncodeCommandPool();
        errdefer {
            if (self.encode_command_pool) |encode_command_pool| {
                self.device.destroyCommandPool(encode_command_pool, null);
            }
        }
        try self.createVideoSession();
        errdefer {
            if (self.video_session) |video_session| {
                self.device.destroyVideoSessionKHR(video_session, null);
            }
        }

        try self.allocateVideoSessionMemory();
        errdefer {
            for (self.encode_session_bind_memory.items) |bind_mem| {
                self.device.freeMemory(bind_mem.memory, null);
            }
        }

        try self.createVideoSessionParameters();
        errdefer self.device.destroyVideoSessionParametersKHR(
            self.video_session_parameters.?,
            null,
        );

        try self.readBitstreamHeader();
        errdefer self.bit_stream_header.deinit();

        try self.allocateOutputBitStream();
        errdefer {
            if (self.bit_stream_memory) |bit_stream_memory| {
                self.device.freeMemory(bit_stream_memory, null);
            }
            if (self.bit_stream_buffer) |bit_stream_buffer| {
                self.device.destroyBuffer(bit_stream_buffer, null);
            }
        }

        try self.allocateReferenceImages(REFERENCE_IMAGE_COUNT);
        errdefer self.destroyImages();

        try self.allocateIntermediateImage();
        errdefer self.destroyIntermediateImages();

        try self.createOutputQueryPool();
        errdefer self.device.destroyQueryPool(self.query_pool.?, null);

        try self.createYCbCrConversionPipeline(self.input_image_views.items);
        errdefer self.destroyYCbCrConversionPipeline();

        try self.createFence();
        errdefer self.destroyEncodeFinishedFence();

        var command_buffer = std.mem.zeroes(vk.CommandBuffer);
        const alloc_info = vk.CommandBufferAllocateInfo{
            .level = .primary,
            .command_pool = self.encode_command_pool.?,
            .command_buffer_count = 1,
        };
        try self.device.allocateCommandBuffers(&alloc_info, @ptrCast(&command_buffer));
        defer self.device.freeCommandBuffers(self.encode_command_pool.?, 1, @ptrCast(&command_buffer));

        const begin_info = vk.CommandBufferBeginInfo{
            .flags = .{ .one_time_submit_bit = true },
        };

        try self.device.beginCommandBuffer(command_buffer, &begin_info);

        self.initRateControl(command_buffer, fps);
        try self.transitionImagesInitial(command_buffer);

        try self.device.endCommandBuffer(command_buffer);
        try self.device.resetFences(1, @ptrCast(&self.encode_finished_fence));
        const submit_info = vk.SubmitInfo{
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&command_buffer),
        };
        try self.device.queueSubmit(
            self.encode_queue,
            1,
            @ptrCast(&submit_info),
            self.encode_finished_fence.?,
        );
        const result = try self.device.waitForFences(
            1,
            @ptrCast(&self.encode_finished_fence),
            vk.TRUE,
            std.math.maxInt(u64),
        );

        if (result != .success) {
            return error.waitForFencesError;
        }
        return self;
    }

    fn createEncodeCommandPool(self: *Self) !void {
        const create_info = vk.CommandPoolCreateInfo{
            .flags = .{ .reset_command_buffer_bit = true },
            .queue_family_index = self.encode_queue_family,
        };
        self.encode_command_pool = try self.device.createCommandPool(&create_info, null);
    }

    fn createVideoSession(self: *Self) !void {
        const video_encode_h264_profile_info = vk.VideoEncodeH264ProfileInfoKHR{ .std_profile_idc = .main };
        self.video_profile = vk.VideoProfileInfoKHR{
            .video_codec_operation = .{ .encode_h264_bit_khr = true },
            .chroma_subsampling = .{ .@"420_bit_khr" = true },
            .chroma_bit_depth = .{ .@"8_bit_khr" = true },
            .luma_bit_depth = .{ .@"8_bit_khr" = true },
            .p_next = &video_encode_h264_profile_info,
            .s_type = .video_profile_info_khr,
        };

        // NOTE: must zero out structs because pointers must be null

        self.video_profile_list = std.mem.zeroes(vk.VideoProfileListInfoKHR);
        self.video_profile_list.?.s_type = .video_profile_list_info_khr;
        self.video_profile_list.?.profile_count = 1;
        self.video_profile_list.?.p_profiles = @ptrCast(&self.video_profile.?);

        var h264_capabilities = std.mem.zeroes(vk.VideoEncodeH264CapabilitiesKHR);
        h264_capabilities.s_type = .video_encode_h264_capabilities_khr;

        var encode_capabilities = std.mem.zeroes(vk.VideoEncodeCapabilitiesKHR);
        encode_capabilities.s_type = .video_encode_capabilities_khr;
        encode_capabilities.p_next = &h264_capabilities;

        var capabilities = std.mem.zeroes(vk.VideoCapabilitiesKHR);
        capabilities.s_type = .video_capabilities_khr;
        capabilities.p_next = &encode_capabilities;

        try self.instance.getPhysicalDeviceVideoCapabilitiesKHR(self.physical_device, &self.video_profile.?, &capabilities);

        self.chosen_rate_control_mode = std.mem.zeroes(vk.VideoEncodeRateControlModeFlagsKHR);
        if (encode_capabilities.rate_control_modes.vbr_bit_khr) {
            self.chosen_rate_control_mode = .{ .vbr_bit_khr = true };
        } else if (encode_capabilities.rate_control_modes.cbr_bit_khr) {
            self.chosen_rate_control_mode = .{ .cbr_bit_khr = true };
        } else if (encode_capabilities.rate_control_modes.disabled_bit_khr) {
            self.chosen_rate_control_mode = .{ .disabled_bit_khr = true };
        }

        var quality_level_info = std.mem.zeroes(vk.PhysicalDeviceVideoEncodeQualityLevelInfoKHR);
        quality_level_info.s_type = .physical_device_video_encode_quality_level_info_khr;
        quality_level_info.p_video_profile = &self.video_profile.?;
        quality_level_info.quality_level = 0;

        var h264_quality_level_properties = std.mem.zeroes(vk.VideoEncodeH264QualityLevelPropertiesKHR);
        h264_quality_level_properties.s_type = .video_encode_h264_quality_level_properties_khr;

        var quality_level_properties = std.mem.zeroes(vk.VideoEncodeQualityLevelPropertiesKHR);
        quality_level_properties.s_type = .video_encode_quality_level_properties_khr;
        quality_level_properties.p_next = &h264_quality_level_properties;

        try self.instance.getPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR(
            self.physical_device,
            &quality_level_info,
            &quality_level_properties,
        );

        var video_format_info = std.mem.zeroes(vk.PhysicalDeviceVideoFormatInfoKHR);
        video_format_info.s_type = .physical_device_video_format_info_khr;
        video_format_info.p_next = &self.video_profile_list;
        video_format_info.image_usage = .{ .video_encode_src_bit_khr = true, .transfer_dst_bit = true };

        var video_format_property_count: u32 = undefined;

        var result = try self.instance.getPhysicalDeviceVideoFormatPropertiesKHR(
            self.physical_device,
            &video_format_info,
            &video_format_property_count,
            null,
        );

        if (result != .success) {
            return error.GetPhysicalDeviceVideoFormatPropertiesKHRResultError;
        }

        var video_format_properties = std.ArrayList(vk.VideoFormatPropertiesKHR).init(self.allocator);
        defer video_format_properties.deinit();

        var video_format_properties_init = std.mem.zeroes(vk.VideoFormatPropertiesKHR);
        video_format_properties_init.s_type = .video_format_properties_khr;
        try video_format_properties.appendNTimes(video_format_properties_init, video_format_property_count);

        result = try self.instance.getPhysicalDeviceVideoFormatPropertiesKHR(
            self.physical_device,
            &video_format_info,
            &video_format_property_count,
            video_format_properties.items.ptr,
        );

        if (result != .success) {
            return error.GetPhysicalDeviceVideoFormatPropertiesKHRResult2Error;
        }

        self.chosen_src_image_format = .undefined;

        for (video_format_properties.items) |vfp| {
            if (vfp.format == .g8_b8r8_2plane_420_unorm or vfp.format == .g8_b8_r8_3plane_420_unorm) {
                // Nvidia driver supports mutable & extended usage, but is not returning those flags
                //constexpr VkImageCreateFlags neededCreateFlags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
                //if ((formatProperties.imageCreateFlags & neededCreateFlags) != neededCreateFlags) {
                //    printf("Skipping format %d, imageCreateFlags not supported\n", formatProperties.format);
                //    continue;
                //}
                self.chosen_src_image_format = vfp.format;
                break;
            }
        }

        if (self.chosen_src_image_format.? == .undefined) {
            return error.InvalidChosenVideoFormat;
        }

        video_format_info.image_usage = std.mem.zeroes(vk.ImageUsageFlags);
        video_format_info.image_usage = .{ .video_encode_dpb_bit_khr = true };

        result = try self.instance.getPhysicalDeviceVideoFormatPropertiesKHR(
            self.physical_device,
            &video_format_info,
            &video_format_property_count,
            null,
        );

        if (result != .success) {
            return error.GetPhysicalDeviceVideoFormatPropertiesKHRResult3Error;
        }

        var dpb_video_format_properties = std.ArrayList(vk.VideoFormatPropertiesKHR).init(self.allocator);
        defer dpb_video_format_properties.deinit();

        var dpb_video_format_properties_init = std.mem.zeroes(vk.VideoFormatPropertiesKHR);
        dpb_video_format_properties_init.s_type = .video_format_properties_khr;
        try dpb_video_format_properties.appendNTimes(dpb_video_format_properties_init, video_format_property_count);

        result = try self.instance.getPhysicalDeviceVideoFormatPropertiesKHR(
            self.physical_device,
            &video_format_info,
            &video_format_property_count,
            dpb_video_format_properties.items.ptr,
        );

        if (result != .success) {
            return error.GetPhysicalDeviceVideoFormatPropertiesKHRResult4Error;
        }

        if (dpb_video_format_properties.items.len < 1) {
            return error.InvalidDpbVideoFormatPropertiesLength;
        }

        self.chosen_dpb_image_format = dpb_video_format_properties.items[0].format;

        var buffer: [256]u8 = std.mem.zeroes([256]u8);
        const my_string: []const u8 = "VK_STD_vulkan_video_codec_h264_encode";
        @memcpy(buffer[0..my_string.len], my_string);

        const h264_std_extension_version = vk.ExtensionProperties{
            .extension_name = buffer,
            .spec_version = @bitCast(vk.makeApiVersion(0, 1, 0, 0)),
        };

        var create_info = std.mem.zeroes(vk.VideoSessionCreateInfoKHR);
        create_info.s_type = .video_session_create_info_khr;
        create_info.queue_family_index = self.encode_queue_family;
        create_info.picture_format = self.chosen_src_image_format.?;
        create_info.max_coded_extent = .{ .width = self.width, .height = self.height };
        create_info.max_dpb_slots = 16;
        create_info.max_active_reference_pictures = 16;
        create_info.reference_picture_format = self.chosen_dpb_image_format.?;
        create_info.p_std_header_version = &h264_std_extension_version;
        create_info.p_video_profile = &self.video_profile.?;

        self.video_session = try self.device.createVideoSessionKHR(&create_info, null);
    }

    fn allocateVideoSessionMemory(self: *Self) !void {
        var video_session_memory_requirement_count: u32 = 0;
        var result = try self.device.getVideoSessionMemoryRequirementsKHR(
            self.video_session.?,
            &video_session_memory_requirement_count,
            null,
        );

        if (result != .success) {
            return error.GetVideoSessionMemoryRequirementsKHRError;
        }

        var encode_session_memory_requirements = std.ArrayList(vk.VideoSessionMemoryRequirementsKHR).init(
            self.allocator,
        );

        var encode_session_memory_requirements_init = std.mem.zeroes(vk.VideoSessionMemoryRequirementsKHR);
        encode_session_memory_requirements_init.s_type = .video_session_memory_requirements_khr;
        try encode_session_memory_requirements.appendNTimes(encode_session_memory_requirements_init, video_session_memory_requirement_count);

        result = try self.device.getVideoSessionMemoryRequirementsKHR(
            self.video_session.?,
            &video_session_memory_requirement_count,
            encode_session_memory_requirements.items.ptr,
        );

        if (result != .success) {
            return error.GetVideoSessionMemoryRequirementsKHRError;
        }

        if (video_session_memory_requirement_count == 0) {
            return error.VideoSessionMemoryRequirementCount0;
        }

        for (0..video_session_memory_requirement_count) |memIdx| {
            const device_memory = try self.vulkan_app.allocate(encode_session_memory_requirements.items[memIdx].memory_requirements, .{});

            var bind_mem = std.mem.zeroes(vk.BindVideoSessionMemoryInfoKHR);
            bind_mem.s_type = .bind_video_session_memory_info_khr;
            bind_mem.memory = device_memory;
            bind_mem.memory_bind_index = encode_session_memory_requirements.items[memIdx].memory_bind_index;
            bind_mem.memory_offset = 0;
            bind_mem.memory_size = encode_session_memory_requirements.items[memIdx].memory_requirements.size;

            try self.encode_session_bind_memory.append(bind_mem);
        }

        try self.device.bindVideoSessionMemoryKHR(self.video_session.?, video_session_memory_requirement_count, self.encode_session_bind_memory.items.ptr);
    }

    fn createVideoSessionParameters(self: *Self) !void {
        self.vui = h264_parameters.getStdVideoH264SequenceParameterSetVui(self.fps);
        self.sps = h264_parameters.getStdVideoH264SequenceParameterSet(self.width, self.height, &self.vui.?);
        self.pps = h264_parameters.getStdVideoH264PictureParameterSet();

        var encode_h264_session_parameters_add_info = std.mem.zeroes(vk.VideoEncodeH264SessionParametersAddInfoKHR);
        encode_h264_session_parameters_add_info.s_type = .video_encode_h264_session_parameters_add_info_khr;
        encode_h264_session_parameters_add_info.std_sps_count = 1;
        encode_h264_session_parameters_add_info.p_std_sp_ss = @ptrCast(&self.sps.?);
        encode_h264_session_parameters_add_info.std_pps_count = 1;
        encode_h264_session_parameters_add_info.p_std_pp_ss = @ptrCast(&self.pps.?);

        var encode_h264_session_parameters_create_info = std.mem.zeroes(vk.VideoEncodeH264SessionParametersCreateInfoKHR);
        encode_h264_session_parameters_create_info.s_type = .video_encode_h264_session_parameters_create_info_khr;
        encode_h264_session_parameters_create_info.max_std_sps_count = 1;
        encode_h264_session_parameters_create_info.max_std_pps_count = 1;
        encode_h264_session_parameters_create_info.p_parameters_add_info = &encode_h264_session_parameters_add_info;

        var session_parameters_create_info = std.mem.zeroes(vk.VideoSessionParametersCreateInfoKHR);
        session_parameters_create_info.s_type = .video_session_parameters_create_info_khr;
        session_parameters_create_info.p_next = &encode_h264_session_parameters_create_info;
        session_parameters_create_info.video_session = self.video_session.?;

        self.video_session_parameters = try self.device.createVideoSessionParametersKHR(
            &session_parameters_create_info,
            null,
        );
    }

    fn readBitstreamHeader(self: *Self) !void {
        var h264_get_info = std.mem.zeroes(vk.VideoEncodeH264SessionParametersGetInfoKHR);
        h264_get_info.s_type = .video_encode_h264_session_parameters_get_info_khr;
        h264_get_info.std_sps_id = 0;
        h264_get_info.std_pps_id = 0;
        h264_get_info.write_std_pps = vk.TRUE;
        h264_get_info.write_std_sps = vk.TRUE;

        var get_info = std.mem.zeroes(vk.VideoEncodeSessionParametersGetInfoKHR);
        get_info.s_type = .video_encode_session_parameters_get_info_khr;
        get_info.p_next = &h264_get_info;
        get_info.video_session_parameters = self.video_session_parameters.?;

        var h264_feedback = std.mem.zeroes(vk.VideoEncodeH264SessionParametersFeedbackInfoKHR);
        h264_feedback.s_type = .video_encode_h264_session_parameters_feedback_info_khr;

        var feedback = std.mem.zeroes(vk.VideoEncodeSessionParametersFeedbackInfoKHR);
        feedback.s_type = .video_encode_session_parameters_feedback_info_khr;
        feedback.p_next = &h264_feedback;

        var datalen: usize = 1024;
        var result = try self.device.getEncodedVideoSessionParametersKHR(&get_info, null, &datalen, null);

        if (result != .success) {
            return error.GetEncodedVideoSessionParametersKHRError;
        }

        try self.bit_stream_header.resize(datalen);

        result = try self.device.getEncodedVideoSessionParametersKHR(
            &get_info,
            &feedback,
            &datalen,
            self.bit_stream_header.items.ptr,
        );

        if (result != .success) {
            return error.GetEncodedVideoSessionParametersKHRError;
        }

        try self.bit_stream_header.resize(datalen);
        self.bit_stream_header_pending = true;
    }

    fn allocateOutputBitStream(self: *Self) !void {
        var buffer_info = std.mem.zeroes(vk.BufferCreateInfo);
        buffer_info.s_type = .buffer_create_info;
        buffer_info.size = 4 * 1024 * 1024;
        buffer_info.usage = .{ .video_encode_dst_bit_khr = true };
        buffer_info.sharing_mode = .exclusive;
        buffer_info.p_next = &self.video_profile_list.?;

        self.bit_stream_buffer = try self.device.createBuffer(&buffer_info, null);
        errdefer self.device.destroyBuffer(self.bit_stream_buffer.?, null);
        const memory_reqs = self.device.getBufferMemoryRequirements(self.bit_stream_buffer.?);

        self.bit_stream_memory = try self.vulkan_app.allocate(memory_reqs, .{
            .host_visible_bit = true,
        });
        errdefer self.device.freeMemory(self.bit_stream_memory.?, null);

        try self.device.bindBufferMemory(self.bit_stream_buffer.?, self.bit_stream_memory.?, 0);

        self.bit_stream_data = @ptrCast(try self.device.mapMemory(self.bit_stream_memory.?, 0, memory_reqs.size, .{}));
    }

    fn allocateReferenceImages(self: *Self, count: u32) !void {
        for (0..count) |_| {
            const image_create_info = vk.ImageCreateInfo{
                .p_next = &self.video_profile_list.?,
                .image_type = .@"2d",
                .format = self.chosen_dpb_image_format.?,
                .extent = .{ .height = self.height, .width = self.width, .depth = 1 },
                .mip_levels = 1,
                .array_layers = 1,
                .samples = .{ .@"1_bit" = true },
                .tiling = .optimal,
                .usage = .{ .video_encode_dpb_bit_khr = true },
                .sharing_mode = .exclusive,
                .queue_family_index_count = 1,
                .p_queue_family_indices = @ptrCast(&self.encode_queue_family),
                .initial_layout = .undefined,
                .flags = .{},
            };

            const image = try self.device.createImage(&image_create_info, null);
            try self.dpb_images.append(image);
            const memory_reqs = self.device.getImageMemoryRequirements(image);
            const memory = try self.vulkan_app.allocate(memory_reqs, .{});
            try self.dpb_image_memory.append(memory);
            try self.device.bindImageMemory(image, memory, 0);

            const view_info = vk.ImageViewCreateInfo{
                .image = image,
                .view_type = .@"2d",
                .format = self.chosen_dpb_image_format.?,
                .subresource_range = .{
                    .aspect_mask = .{ .color_bit = true },
                    .base_mip_level = 0,
                    .level_count = 1,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
                .components = std.mem.zeroes(vk.ComponentMapping),
            };
            const image_view = try self.device.createImageView(&view_info, null);
            try self.dpb_image_views.append(image_view);
        }
    }

    fn allocateIntermediateImage(self: *Self) !void {
        var image_create_info = vk.ImageCreateInfo{
            .p_next = &self.video_profile_list.?,
            .image_type = .@"2d",
            .format = self.chosen_src_image_format.?,
            .extent = .{ .height = self.height, .width = self.width, .depth = 1 },
            .mip_levels = 1,
            .array_layers = 1,
            .samples = .{ .@"1_bit" = true },
            .tiling = .optimal,
            .usage = .{ .video_encode_src_bit_khr = true, .storage_bit = true },
            .sharing_mode = .exclusive,
            .queue_family_index_count = 0,
            .p_queue_family_indices = null,
            .initial_layout = .undefined,
            .flags = .{ .mutable_format_bit = true, .extended_usage_bit = true },
        };

        const queue_families = [_]u32{ self.encode_queue_family, self.compute_queue_family };

        if (self.encode_queue_family != self.compute_queue_family) {
            image_create_info.sharing_mode = .concurrent;
            image_create_info.queue_family_index_count = 2;
            image_create_info.p_queue_family_indices = &queue_families;
        }

        self.ycbcr_image = try self.device.createImage(&image_create_info, null);
        const memory_reqs = self.device.getImageMemoryRequirements(self.ycbcr_image.?);
        self.ycbcr_image_memory = try self.vulkan_app.allocate(memory_reqs, .{ .device_local_bit = true });
        try self.device.bindImageMemory(self.ycbcr_image.?, self.ycbcr_image_memory.?, 0);

        var view_usage_info = vk.ImageViewUsageCreateInfo{
            .usage = .{ .video_encode_src_bit_khr = true },
        };

        var view_info = vk.ImageViewCreateInfo{
            .p_next = &view_usage_info,
            .image = self.ycbcr_image.?,
            .view_type = .@"2d",
            .format = self.chosen_src_image_format.?,
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
            .components = std.mem.zeroes(vk.ComponentMapping),
        };

        self.ycbcr_image_view = try self.device.createImageView(&view_info, null);

        view_usage_info.usage = .{ .storage_bit = true };
        const num_planes: u32 = if (self.chosen_src_image_format == .g8_b8r8_2plane_420_unorm) 2 else 3;

        view_info.format = .r8_unorm;
        view_info.subresource_range.aspect_mask = .{ .plane_0_bit = true };
        try self.ycbcr_image_plane_views.append(try self.device.createImageView(&view_info, null));

        view_info.subresource_range.aspect_mask = .{ .plane_1_bit = true };

        if (num_planes == 2) {
            view_info.format = .r8g8_unorm;
            try self.ycbcr_image_plane_views.append(try self.device.createImageView(&view_info, null));
        } else {
            try self.ycbcr_image_plane_views.append(try self.device.createImageView(&view_info, null));
            view_info.subresource_range.aspect_mask = .{ .plane_2_bit = true };
            try self.ycbcr_image_plane_views.append(try self.device.createImageView(&view_info, null));
        }
    }

    fn createOutputQueryPool(self: *Self) !void {
        const query_pool_video_encode_feedback_create_info = vk.QueryPoolVideoEncodeFeedbackCreateInfoKHR{
            .p_next = &self.video_profile.?,
            .encode_feedback_flags = .{
                .bitstream_buffer_offset_bit_khr = true,
                .bitstream_bytes_written_bit_khr = true,
            },
        };

        const query_pool_create_info = vk.QueryPoolCreateInfo{
            .query_type = .video_encode_feedback_khr,
            .query_count = 1,
            .p_next = &query_pool_video_encode_feedback_create_info,
        };

        self.query_pool = try self.device.createQueryPool(&query_pool_create_info, null);
    }

    fn createYCbCrConversionPipeline(self: *Self, image_views: []vk.ImageView) !void {
        _ = image_views;
        const rgb_ycbcr_shader_2plane = @embedFile("rgb-ycbcr-shader-2plane");
        const rgb_ycbcr_shader_3plane = @embedFile("rgb-ycbcr-shader-3plane");
        const compute_shader = if (self.ycbcr_image_plane_views.items.len == 2) rgb_ycbcr_shader_2plane else rgb_ycbcr_shader_3plane;
        const shader_module_create_info = vk.ShaderModuleCreateInfo{
            .code_size = compute_shader.len,
            .p_code = @alignCast(@ptrCast(compute_shader)),
        };
        const compute_shader_module = try self.device.createShaderModule(&shader_module_create_info, null);
        defer self.device.destroyShaderModule(compute_shader_module, null);

        const compute_shader_stage_info = vk.PipelineShaderStageCreateInfo{
            .stage = .{ .compute_bit = true },
            .module = compute_shader_module,
            .p_name = "main",
        };

        var layout_bindings = std.mem.zeroes([4]vk.DescriptorSetLayoutBinding);

        for (&layout_bindings, 0..) |*lb, i| {
            lb.binding = @intCast(i);
            lb.descriptor_count = 1;
            lb.descriptor_type = .storage_image;
            lb.stage_flags = .{ .compute_bit = true };
        }

        const ycbcr_image_plane_view_size: u32 = @intCast(self.ycbcr_image_plane_views.items.len);
        const layout_info = vk.DescriptorSetLayoutCreateInfo{
            .binding_count = 1 + ycbcr_image_plane_view_size,
            .p_bindings = &layout_bindings,
        };

        self.compute_descriptor_set_layout = try self.device.createDescriptorSetLayout(&layout_info, null);

        const pipeline_layout_create_info = vk.PipelineLayoutCreateInfo{
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&self.compute_descriptor_set_layout.?),
        };

        self.compute_pipeline_layout = try self.device.createPipelineLayout(&pipeline_layout_create_info, null);

        const compute_pipeline_info = vk.ComputePipelineCreateInfo{
            .layout = self.compute_pipeline_layout.?,
            .stage = compute_shader_stage_info,
            .base_pipeline_index = 0,
        };

        var compute_pipeline: [1]vk.Pipeline = [_]vk.Pipeline{std.mem.zeroes(vk.Pipeline)};
        const result = try self.device.createComputePipelines(
            .null_handle,
            1,
            @ptrCast(&compute_pipeline_info),
            null,
            &compute_pipeline,
        );

        if (result != .success) {
            return error.createComputePipelinesError;
        }

        self.compute_pipeline = compute_pipeline[0];

        const max_frames_count: u32 = @intCast(self.input_image_views.items.len);
        var pool_sizes: [1]vk.DescriptorPoolSize = std.mem.zeroes([1]vk.DescriptorPoolSize);
        pool_sizes[0].descriptor_count = 4 * max_frames_count;

        const pool_info = vk.DescriptorPoolCreateInfo{
            .pool_size_count = pool_sizes.len,
            .p_pool_sizes = pool_sizes[0..].ptr,
            .max_sets = max_frames_count,
        };

        self.descriptor_pool = try self.device.createDescriptorPool(&pool_info, null);

        var layouts = std.ArrayList(vk.DescriptorSetLayout).init(self.allocator);
        defer layouts.deinit();
        try layouts.resize(max_frames_count);

        for (layouts.items) |*dsl| {
            dsl.* = self.compute_descriptor_set_layout.?;
        }

        const descriptor_set_alloc_info = vk.DescriptorSetAllocateInfo{
            .descriptor_pool = self.descriptor_pool.?,
            .descriptor_set_count = max_frames_count,
            .p_set_layouts = layouts.items.ptr,
        };

        try self.compute_descriptor_sets.resize(max_frames_count);
        try self.device.allocateDescriptorSets(&descriptor_set_alloc_info, self.compute_descriptor_sets.items.ptr);

        for (0..max_frames_count) |i| {
            var descriptor_writes: [4]vk.WriteDescriptorSet = undefined;
            // NOTE: need to zero these out
            for (&descriptor_writes) |*d| {
                d.* = std.mem.zeroes(vk.WriteDescriptorSet);
            }
            var image_infos: [4]vk.DescriptorImageInfo = undefined;

            image_infos[0].image_view = self.input_image_views.items[i];
            image_infos[0].image_layout = .general;
            image_infos[0].sampler = .null_handle;
            descriptor_writes[0].s_type = .write_descriptor_set;
            descriptor_writes[0].dst_set = self.compute_descriptor_sets.items[i];
            descriptor_writes[0].dst_binding = 0;
            descriptor_writes[0].dst_array_element = 0;
            descriptor_writes[0].descriptor_type = .storage_image;
            descriptor_writes[0].descriptor_count = 1;
            descriptor_writes[0].p_image_info = @ptrCast(&image_infos[0]);

            for (0..self.ycbcr_image_plane_views.items.len) |p| {
                image_infos[p + 1].image_view = self.ycbcr_image_plane_views.items[p];
                image_infos[p + 1].image_layout = .general;
                image_infos[p + 1].sampler = .null_handle;
                descriptor_writes[p + 1].s_type = .write_descriptor_set;
                descriptor_writes[p + 1].dst_set = self.compute_descriptor_sets.items[i];
                descriptor_writes[p + 1].dst_binding = @as(u32, @intCast(p)) + 1;
                descriptor_writes[p + 1].dst_array_element = 0;
                descriptor_writes[p + 1].descriptor_type = .storage_image;
                descriptor_writes[p + 1].descriptor_count = 1;
                descriptor_writes[p + 1].p_image_info = @ptrCast(&image_infos[p + 1]);
            }

            self.device.updateDescriptorSets(
                @as(u32, @intCast(self.ycbcr_image_plane_views.items.len)) + 1,
                descriptor_writes[0..].ptr,
                0,
                null,
            );
        }
    }

    fn createFence(self: *Self) !void {
        self.inter_queue_semaphore1 = try self.device.createSemaphore(&.{}, null);
        self.inter_queue_semaphore2 = try self.device.createSemaphore(&.{}, null);
        self.inter_queue_semaphore3 = try self.device.createSemaphore(&.{}, null);

        self.encode_finished_fence = try self.device.createFence(&.{ .flags = .{ .signaled_bit = true } }, null);
    }

    fn initRateControl(self: *Self, command_buffer: vk.CommandBuffer, fps: u32) void {
        self.encode_h264_rate_control_layer_info = std.mem.zeroes(vk.VideoEncodeH264RateControlLayerInfoKHR);
        self.encode_h264_rate_control_layer_info.?.s_type = .video_encode_h264_rate_control_layer_info_khr;

        self.encode_rate_control_layer_info = vk.VideoEncodeRateControlLayerInfoKHR{
            .p_next = &self.encode_h264_rate_control_layer_info.?,
            .frame_rate_numerator = fps,
            .frame_rate_denominator = 1,
            .average_bitrate = 5_000_000,
            .max_bitrate = 20_000_000,
        };

        self.encode_h264_rate_control_info = vk.VideoEncodeH264RateControlInfoKHR{
            .flags = .{ .regular_gop_bit_khr = true, .reference_pattern_flat_bit_khr = true },
            .gop_frame_count = 16,
            .idr_period = 16,
            .consecutive_b_frame_count = 0,
            .temporal_layer_count = 1,
        };

        self.encode_rate_control_info = vk.VideoEncodeRateControlInfoKHR{
            .p_next = &self.encode_h264_rate_control_info.?,
            .rate_control_mode = self.chosen_rate_control_mode.?,
            .layer_count = 1,
            .p_layers = @ptrCast(&self.encode_rate_control_layer_info.?),
            .initial_virtual_buffer_size_in_ms = 100,
            .virtual_buffer_size_in_ms = 200,
        };

        if (self.encode_rate_control_info.?.rate_control_mode.cbr_bit_khr) {
            self.encode_rate_control_layer_info.?.average_bitrate = self.encode_rate_control_layer_info.?.max_bitrate;
        }

        if (self.encode_rate_control_info.?.rate_control_mode.disabled_bit_khr or @as(
            u32,
            @bitCast(self.encode_rate_control_info.?.rate_control_mode),
        ) == 0) {
            self.encode_h264_rate_control_info.?.temporal_layer_count = 0;
            self.encode_rate_control_info.?.layer_count = 0;
        }

        const begin_coding_info = vk.VideoBeginCodingInfoKHR{
            .video_session = self.video_session.?,
            .video_session_parameters = self.video_session_parameters.?,
        };
        self.device.cmdBeginVideoCodingKHR(command_buffer, &begin_coding_info);

        const coding_control_info = vk.VideoCodingControlInfoKHR{
            .flags = .{ .reset_bit_khr = true, .encode_rate_control_bit_khr = true },
            .p_next = &self.encode_rate_control_info,
        };
        self.device.cmdControlVideoCodingKHR(command_buffer, &coding_control_info);
        self.device.cmdEndVideoCodingKHR(command_buffer, &.{});
    }

    fn transitionImagesInitial(self: *Self, command_buffer: vk.CommandBuffer) !void {
        var barriers = std.ArrayList(vk.ImageMemoryBarrier2).init(self.allocator);
        defer barriers.deinit();

        for (self.dpb_images.items) |dpb_image| {
            const image_memory_barrier = vk.ImageMemoryBarrier2{
                .src_stage_mask = .{ .bottom_of_pipe_bit = true },
                .dst_stage_mask = .{ .top_of_pipe_bit = true },
                .old_layout = .undefined,
                .subresource_range = .{
                    .aspect_mask = .{ .color_bit = true },
                    .base_mip_level = 0,
                    .level_count = 1,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
                .image = dpb_image,
                .new_layout = .video_encode_dpb_khr,
                .src_queue_family_index = 0,
                .dst_queue_family_index = 0,
            };
            try barriers.append(image_memory_barrier);
        }

        const dependency_info = vk.DependencyInfoKHR{
            .image_memory_barrier_count = @as(u32, @intCast(barriers.items.len)),
            .p_image_memory_barriers = barriers.items.ptr,
        };

        self.device.cmdPipelineBarrier2(command_buffer, &dependency_info);
    }

    pub fn finishEncode(self: *Self, data: *?[*]u8, size: *usize) !void {
        if (!self.running) {
            size.* = 0;
            return;
        }

        if (self.bit_stream_header_pending) {
            data.* = self.bit_stream_header.items.ptr;
            size.* = self.bit_stream_header.items.len;
            self.bit_stream_header_pending = false;
            return;
        }

        try self.getOutputVideoPacket(data, size);

        if (self.compute_command_buffer) |compute_command_buffer| {
            self.device.freeCommandBuffers(self.compute_command_pool, 1, @ptrCast(&compute_command_buffer));
        }

        if (self.encode_command_buffer) |encode_command_buffer| {
            self.device.freeCommandBuffers(self.encode_command_pool.?, 1, @ptrCast(&encode_command_buffer));
        }

        self.frame_count += 1;
        self.running = false;
    }

    fn getOutputVideoPacket(self: *Self, data: *?[*]u8, size: *usize) !void {
        var result = try self.device.waitForFences(
            1,
            @ptrCast(&self.encode_finished_fence.?),
            vk.TRUE,
            std.math.maxInt(u64),
        );

        if (result != .success) {
            return error.waitForFencesError;
        }

        const VideoEncodeStatus = extern struct {
            bitstream_start_offset: u32,
            bitstream_size: u32,
            status: vk.QueryResultStatusKHR,
        };

        var encode_result = std.mem.zeroes(VideoEncodeStatus);
        const query_slot_id: u32 = 0;
        result = try self.device.getQueryPoolResults(
            self.query_pool.?,
            query_slot_id,
            1,
            @sizeOf(VideoEncodeStatus),
            @ptrCast(&encode_result),
            @sizeOf(VideoEncodeStatus),
            .{ .with_status_bit_khr = true, .wait_bit = true },
        );

        if (result != .success) {
            return error.getQueryPoolResultsError;
        }

        const mapped_memory_range = vk.MappedMemoryRange{
            .memory = self.bit_stream_memory.?,
            .offset = encode_result.bitstream_start_offset,
            .size = encode_result.bitstream_size,
        };
        try self.device.invalidateMappedMemoryRanges(1, @ptrCast(&mapped_memory_range));

        const d = @as(
            [*]u8,
            @ptrFromInt(@intFromPtr(self.bit_stream_data.?) + encode_result.bitstream_start_offset),
        );
        data.* = d;
        size.* = encode_result.bitstream_size;

        print("Encoded frame {}, status {}, offset {}, size {}\n", .{
            self.frame_count,
            encode_result.status,
            encode_result.bitstream_start_offset,
            encode_result.bitstream_size,
        });
    }

    pub fn queueEncode(self: *Self, current_image_ix: u32) !void {
        try self.convertRGBtoYCbCr(current_image_ix);
        try self.encodeVideoFrame();
        self.running = true;
    }

    fn convertRGBtoYCbCr(self: *Self, current_image_ix: u32) !void {
        const alloc_info = vk.CommandBufferAllocateInfo{
            .level = .primary,
            .command_pool = self.compute_command_pool,
            .command_buffer_count = 1,
        };

        var command_buffer = std.mem.zeroes(vk.CommandBuffer);
        try self.device.allocateCommandBuffers(&alloc_info, @ptrCast(&command_buffer));
        self.compute_command_buffer = command_buffer;

        try self.device.beginCommandBuffer(self.compute_command_buffer.?, &.{
            .flags = .{ .one_time_submit_bit = true },
        });

        var barriers = std.ArrayList(vk.ImageMemoryBarrier2).init(self.allocator);
        defer barriers.deinit();
        var image_memory_barrier = vk.ImageMemoryBarrier2{
            .dst_stage_mask = .{ .compute_shader_bit = true },
            .new_layout = .general,
            .subresource_range = .{
                .aspect_mask = .{
                    .plane_0_bit = true,
                    .plane_1_bit = true,
                },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
            // transition YCbCr image (luma and chroma planes) to be shader target
            .src_stage_mask = .{},
            .src_access_mask = .{},
            .old_layout = .undefined,
            .image = self.ycbcr_image.?,
            .dst_access_mask = .{ .shader_storage_write_bit = true },
            .src_queue_family_index = 0,
            .dst_queue_family_index = 0,
        };

        if (self.ycbcr_image_plane_views.items.len >= 3) {
            image_memory_barrier.subresource_range.aspect_mask.plane_2_bit = true;
        }
        try barriers.append(image_memory_barrier);

        // transition source image to be shader source
        image_memory_barrier.src_stage_mask = .{ .color_attachment_output_bit = true };
        image_memory_barrier.src_access_mask = .{ .color_attachment_write_bit = true };
        image_memory_barrier.old_layout = .color_attachment_optimal;
        image_memory_barrier.image = self.input_images.items[current_image_ix];
        image_memory_barrier.dst_access_mask = .{ .shader_storage_read_bit = true };
        image_memory_barrier.subresource_range.aspect_mask = .{ .color_bit = true };
        try barriers.append(image_memory_barrier);

        const dependency_info = vk.DependencyInfo{
            .image_memory_barrier_count = @as(u32, @intCast(barriers.items.len)),
            .p_image_memory_barriers = barriers.items.ptr,
        };

        self.device.cmdPipelineBarrier2(self.compute_command_buffer.?, &dependency_info);

        // run the RGB->YCbCr conversion shader
        self.device.cmdBindPipeline(self.compute_command_buffer.?, .compute, self.compute_pipeline.?);
        self.device.cmdBindDescriptorSets(
            self.compute_command_buffer.?,
            .compute,
            self.compute_pipeline_layout.?,
            0,
            1,
            @ptrCast(&self.compute_descriptor_sets.items[current_image_ix]),
            0,
            null,
        );

        // work item local size = 16x16
        self.device.cmdDispatch(
            self.compute_command_buffer.?,
            (self.width + 15) / 16,
            (self.height + 15) / 16,
            1,
        );

        try self.device.endCommandBuffer(self.compute_command_buffer.?);

        const dst_stage_masks: [2]vk.PipelineStageFlags = .{ .{ .all_commands_bit = true }, .{ .all_commands_bit = true } };
        const signal_semaphores: [2]vk.Semaphore = .{ self.inter_queue_semaphore1.?, self.inter_queue_semaphore3.? };
        const wait_semaphores: [2]vk.Semaphore = .{ self.inter_queue_semaphore2.?, self.inter_queue_semaphore3.? };
        var submit_info = vk.SubmitInfo{
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&self.compute_command_buffer.?),
            .signal_semaphore_count = 2,
            .p_signal_semaphores = &signal_semaphores,
        };

        if (self.frame_count != 0) {
            submit_info.wait_semaphore_count = 2;
            submit_info.p_wait_semaphores = &wait_semaphores;
            submit_info.p_wait_dst_stage_mask = &dst_stage_masks;
        }

        try self.device.queueSubmit(self.compute_queue, 1, @ptrCast(&submit_info), .null_handle);
    }

    fn encodeVideoFrame(self: *Self) !void {
        const GOP_LENGTH: u32 = 16;
        const gop_frame_count = self.frame_count % GOP_LENGTH;

        const alloc_info = vk.CommandBufferAllocateInfo{
            .level = .primary,
            .command_pool = self.encode_command_pool.?,
            .command_buffer_count = 1,
        };

        var command_buffer = std.mem.zeroes(vk.CommandBuffer);
        try self.device.allocateCommandBuffers(&alloc_info, @ptrCast(&command_buffer));
        self.encode_command_buffer = command_buffer;

        const begin_info = vk.CommandBufferBeginInfo{
            .flags = .{ .one_time_submit_bit = true },
        };

        try self.device.beginCommandBuffer(self.encode_command_buffer.?, &begin_info);

        const query_slot_id = 0;
        self.device.cmdResetQueryPool(self.encode_command_buffer.?, self.query_pool.?, query_slot_id, 1);

        // start a video encode session
        // set an image view as DPB (decoded output picture)
        const dpb_pic_resource = vk.VideoPictureResourceInfoKHR{
            .image_view_binding = self.dpb_image_views.items[gop_frame_count & 1],
            .coded_offset = .{ .x = 0, .y = 0 },
            .coded_extent = .{ .height = self.height, .width = self.width },
            .base_array_layer = 0,
        };

        // set an image view as reference picture
        const ref_pic_resource = vk.VideoPictureResourceInfoKHR{
            .image_view_binding = self.dpb_image_views.items[@intFromBool((gop_frame_count & 1) == 0)],
            .coded_offset = .{ .x = 0, .y = 0 },
            .coded_extent = .{ .height = self.height, .width = self.width },
            .base_array_layer = 0,
        };

        const max_pic_order_cnt_lsb: u32 = @as(u32, 1) << @as(u5, @intCast(self.sps.?.log_2_max_pic_order_cnt_lsb_minus_4 + 4));

        var dpb_ref_info = std.mem.zeroes(vk.StdVideoEncodeH264ReferenceInfo);
        dpb_ref_info.frame_num = gop_frame_count;
        dpb_ref_info.pic_order_cnt = @as(i32, @intCast((dpb_ref_info.frame_num * 2) % max_pic_order_cnt_lsb));
        dpb_ref_info.primary_pic_type = if (dpb_ref_info.frame_num == 0) .idr else .p;

        const dpb_slot_info = vk.VideoEncodeH264DpbSlotInfoKHR{
            .p_std_reference_info = &dpb_ref_info,
        };

        var ref_ref_info = std.mem.zeroes(vk.StdVideoEncodeH264ReferenceInfo);
        ref_ref_info.frame_num = @subWithOverflow(gop_frame_count, 1)[0];
        ref_ref_info.pic_order_cnt = @as(i32, @intCast((@mulWithOverflow(ref_ref_info.frame_num, 2)[0]) % max_pic_order_cnt_lsb));
        ref_ref_info.primary_pic_type = if (ref_ref_info.frame_num == 0) .idr else .p;

        const ref_slot_info = vk.VideoEncodeH264DpbSlotInfoKHR{
            .p_std_reference_info = &ref_ref_info,
        };

        var referense_slots = std.mem.zeroes([2]vk.VideoReferenceSlotInfoKHR);
        referense_slots[0].s_type = .video_reference_slot_info_khr;
        referense_slots[0].p_next = &dpb_slot_info;
        referense_slots[0].slot_index = -1;
        referense_slots[0].p_picture_resource = &dpb_pic_resource;
        referense_slots[1].s_type = .video_reference_slot_info_khr;
        referense_slots[1].p_next = &ref_slot_info;
        referense_slots[1].slot_index = @intFromBool((gop_frame_count & 1) == 0);
        referense_slots[1].p_picture_resource = &ref_pic_resource;

        const encode_begin_info = vk.VideoBeginCodingInfoKHR{
            .p_next = &self.encode_rate_control_info.?,
            .video_session = self.video_session.?,
            .video_session_parameters = self.video_session_parameters.?,
            .reference_slot_count = if (gop_frame_count == 0) 1 else 2,
            .p_reference_slots = &referense_slots,
        };

        self.device.cmdBeginVideoCodingKHR(self.encode_command_buffer.?, &encode_begin_info);

        // transition the YCbCr image to be a video encode source
        const image_memory_barrier = vk.ImageMemoryBarrier2{
            .src_stage_mask = .{},
            .src_access_mask = .{},
            .dst_stage_mask = .{ .video_encode_bit_khr = true },
            .dst_access_mask = .{ .video_encode_read_bit_khr = true },
            .old_layout = .general,
            .new_layout = .video_encode_src_khr,
            .image = self.ycbcr_image.?,
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
            .src_queue_family_index = 0,
            .dst_queue_family_index = 0,
        };

        const dependency_info = vk.DependencyInfoKHR{
            .image_memory_barrier_count = 1,
            .p_image_memory_barriers = @ptrCast(&image_memory_barrier),
        };

        self.device.cmdPipelineBarrier2(self.encode_command_buffer.?, &dependency_info);

        const input_pic_resource = vk.VideoPictureResourceInfoKHR{
            .image_view_binding = self.ycbcr_image_view.?,
            .coded_offset = .{ .x = 0, .y = 0 },
            .coded_extent = .{ .height = self.height, .width = self.width },
            .base_array_layer = 0,
        };

        var frame_info = h264_parameters.FrameInfo{};
        h264_parameters.FrameInfo.init(
            &frame_info,
            gop_frame_count,
            self.width,
            self.height,
            self.sps.?,
            self.pps.?,
            gop_frame_count,
            self.chosen_rate_control_mode.?.disabled_bit_khr,
        );

        const encode_h262_frame_info = frame_info.getEncodeH264FrameInfo();

        var video_encode_info = vk.VideoEncodeInfoKHR{
            .p_next = encode_h262_frame_info,
            .dst_buffer = self.bit_stream_buffer.?,
            .dst_buffer_offset = 0,
            .dst_buffer_range = 4 * 1024 * 1024,
            .src_picture_resource = input_pic_resource,
            .preceding_externally_encoded_bytes = 0,
        };

        referense_slots[0].slot_index = @as(i32, @intCast(gop_frame_count & 1));
        video_encode_info.p_setup_reference_slot = &referense_slots[0];

        if (gop_frame_count > 0) {
            video_encode_info.reference_slot_count = 1;
            video_encode_info.p_reference_slots = @ptrCast(&referense_slots[1]);
        }

        // prepare the query pool for the resulting bitstream
        self.device.cmdBeginQuery(self.encode_command_buffer.?, self.query_pool.?, query_slot_id, .{});
        // encode the frame as video
        self.device.cmdEncodeVideoKHR(self.encode_command_buffer.?, &video_encode_info);
        // end the query for the result
        self.device.cmdEndQuery(self.encode_command_buffer.?, self.query_pool.?, query_slot_id);
        // finish the video session
        self.device.cmdEndVideoCodingKHR(self.encode_command_buffer.?, &.{});

        // run the encoding
        try self.device.endCommandBuffer(self.encode_command_buffer.?);

        const dst_stage_mask = vk.PipelineStageFlags{
            .all_commands_bit = true,
        };
        const submit_info = vk.SubmitInfo{
            .wait_semaphore_count = 1,
            .p_wait_semaphores = @ptrCast(&self.inter_queue_semaphore1),
            .p_wait_dst_stage_mask = @ptrCast(&dst_stage_mask),
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&self.encode_command_buffer.?),
            .signal_semaphore_count = 1,
            .p_signal_semaphores = @ptrCast(&self.inter_queue_semaphore2.?),
        };
        try self.device.resetFences(1, @ptrCast(&self.encode_finished_fence.?));
        try self.device.queueSubmit(self.encode_queue, 1, @ptrCast(&submit_info), self.encode_finished_fence.?);
    }

    fn destroyEncodeFinishedFence(self: *Self) void {
        if (self.encode_finished_fence) |f| {
            self.device.destroyFence(f, null);
        }
        if (self.inter_queue_semaphore1) |s| {
            self.device.destroySemaphore(s, null);
        }
        if (self.inter_queue_semaphore2) |s| {
            self.device.destroySemaphore(s, null);
        }
        if (self.inter_queue_semaphore3) |s| {
            self.device.destroySemaphore(s, null);
        }
    }

    fn destroyYCbCrConversionPipeline(self: *Self) void {
        if (self.compute_pipeline) |compute_pipeline| {
            self.device.destroyPipeline(compute_pipeline, null);
        }
        if (self.compute_pipeline_layout) |compute_pipeline_layout| {
            self.device.destroyPipelineLayout(compute_pipeline_layout, null);
        }
        if (self.descriptor_pool) |descriptor_pool| {
            self.device.destroyDescriptorPool(descriptor_pool, null);
        }
        if (self.compute_descriptor_set_layout) |compute_descriptor_set_layout| {
            self.device.destroyDescriptorSetLayout(compute_descriptor_set_layout, null);
        }
    }

    fn destroyImages(self: *Self) void {
        for (self.dpb_images.items) |dpb_image| {
            self.device.destroyImage(dpb_image, null);
        }
        for (self.dpb_image_views.items) |dpb_image_view| {
            self.device.destroyImageView(dpb_image_view, null);
        }
        for (self.dpb_image_memory.items) |dpb_image_memory| {
            self.device.freeMemory(dpb_image_memory, null);
        }
    }

    fn destroyIntermediateImages(self: *Self) void {
        if (self.ycbcr_image) |ycbcr_image| {
            self.device.destroyImage(ycbcr_image, null);
        }

        if (self.ycbcr_image_view) |ycbcr_image_view| {
            self.device.destroyImageView(ycbcr_image_view, null);
        }

        if (self.ycbcr_image_memory) |ycbcr_image_memory| {
            self.device.freeMemory(ycbcr_image_memory, null);
        }

        for (self.ycbcr_image_plane_views.items) |ycbcr_image_plane_view| {
            self.device.destroyImageView(ycbcr_image_plane_view, null);
        }
    }

    pub fn deinit(self: *Self) void {
        if (self.running) {
            var size: usize = 0;
            var data: ?[*]u8 = null;

            self.getOutputVideoPacket(&data, &size) catch |err| {
                std.debug.print("getOutputVideoPacket error: {}", .{err});
            };

            if (self.compute_command_buffer) |compute_command_buffer| {
                self.device.freeCommandBuffers(self.compute_command_pool, 1, @ptrCast(&compute_command_buffer));
            }

            if (self.encode_command_buffer) |encode_command_buffer| {
                self.device.freeCommandBuffers(self.encode_command_pool.?, 1, @ptrCast(&encode_command_buffer));
            }
        }

        self.destroyEncodeFinishedFence();
        self.destroyYCbCrConversionPipeline();
        if (self.video_session_parameters) |video_session_parameters| {
            self.device.destroyVideoSessionParametersKHR(video_session_parameters, null);
        }
        if (self.query_pool) |query_pool| {
            self.device.destroyQueryPool(query_pool, null);
        }

        if (self.bit_stream_memory) |bit_stream_memory| {
            self.device.unmapMemory(bit_stream_memory);
        }

        if (self.bit_stream_buffer) |bit_stream_buffer| {
            self.device.destroyBuffer(bit_stream_buffer, null);
        }

        self.destroyImages();
        self.destroyIntermediateImages();

        if (self.bit_stream_memory) |bit_stream_memory| {
            self.device.freeMemory(bit_stream_memory, null);
        }
        if (self.video_session) |video_session| {
            self.device.destroyVideoSessionKHR(video_session, null);
        }
        for (self.encode_session_bind_memory.items) |bind_mem| {
            self.device.freeMemory(bind_mem.memory, null);
        }
        if (self.encode_command_pool) |encode_command_pool| {
            self.device.destroyCommandPool(encode_command_pool, null);
        }

        self.bit_stream_header.deinit();
        self.encode_session_bind_memory.deinit();
        self.dpb_images.deinit();
        self.dpb_image_memory.deinit();
        self.dpb_image_views.deinit();
        self.ycbcr_image_plane_views.deinit();
        self.compute_descriptor_sets.deinit();
        self.allocator.destroy(self);
    }
};
