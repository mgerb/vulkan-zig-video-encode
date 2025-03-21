const std = @import("std");
const print = std.debug.print;
const vk = @import("vulkan");
const VideoEncoder = @import("./video_encoder.zig").VideoEncoder;

const DEBUG = true;
const FPS = 30;
const TOTAL_FRAMES = 300;
const IMAGE_COUNT = 2;

const DEVICE_EXTENSIONS = [_][*:0]const u8{
    vk.extensions.khr_dynamic_rendering.name,
    vk.extensions.khr_video_queue.name,
    vk.extensions.khr_video_encode_queue.name,
    vk.extensions.khr_video_encode_h_264.name,
    vk.extensions.khr_synchronization_2.name,
};

// C vulkan libs
pub extern fn vkGetInstanceProcAddr(instance: vk.Instance, procname: [*:0]const u8) vk.PfnVoidFunction;

const QueueAllocation = struct {
    graphics_family: u32,
    video_encode_family: u32,
};

const DeviceCandidate = struct {
    pdev: vk.PhysicalDevice,
    props: vk.PhysicalDeviceProperties,
    queues: QueueAllocation,
};

pub const Queue = struct {
    handle: vk.Queue,
    family: u32,

    fn init(device: Device, family: u32) Queue {
        return .{
            .handle = device.getDeviceQueue(family, 0),
            .family = family,
        };
    }
};

fn apis() []const vk.ApiInfo {
    var _apis: []const vk.ApiInfo = &.{
        .{
            .base_commands = .{
                .createInstance = true,
            },
            .instance_commands = .{
                .createDevice = true,
            },
        },
        vk.features.version_1_0,
        vk.features.version_1_3,
        vk.extensions.khr_video_queue,
        vk.extensions.khr_video_encode_queue,
        vk.extensions.khr_video_encode_av_1,
        vk.extensions.khr_video_encode_h_264,
        vk.extensions.khr_video_encode_h_265,
        vk.extensions.khr_video_encode_quantization_map,
        vk.extensions.khr_video_maintenance_1,
        vk.extensions.khr_video_maintenance_2,
        vk.extensions.khr_dynamic_rendering,
        vk.extensions.khr_synchronization_2,
    };

    if (DEBUG) {
        const debug_apis: []const vk.ApiInfo = &.{
            vk.extensions.ext_debug_utils,
        };
        _apis = _apis ++ debug_apis;
    }

    return _apis;
}

const BaseDispatch = vk.BaseWrapper(apis());
const InstanceDispatch = vk.InstanceWrapper(apis());
const DeviceDispatch = vk.DeviceWrapper(apis());

pub const Instance = vk.InstanceProxy(apis());
pub const Device = vk.DeviceProxy(apis());

pub const CommandBuffer = vk.CommandBufferProxy(apis());

pub const VulkanApp = struct {
    const Self = @This();
    allocator: std.mem.Allocator,
    vkb: BaseDispatch,
    instance: Instance,
    device: Device,
    debug_messenger: ?vk.DebugUtilsMessengerEXT,
    graphics_queue: Queue,
    video_encode_queue: Queue,
    physical_device: vk.PhysicalDevice,
    props: vk.PhysicalDeviceProperties,
    mem_props: vk.PhysicalDeviceMemoryProperties,
    images: std.ArrayList(vk.Image),
    image_views: std.ArrayList(vk.ImageView),
    height: u32,
    width: u32,
    image_memory: std.ArrayList(vk.DeviceMemory),

    // graphics pipeline
    graphics_pipeline_layout: vk.PipelineLayout,
    graphics_pipeline: vk.Pipeline,

    command_pool: vk.CommandPool,
    command_buffers: std.ArrayList(vk.CommandBuffer),

    video_encoder: *VideoEncoder = undefined,
    outfile: ?std.fs.File = null,

    pub fn init(allocator: std.mem.Allocator, width: u32, height: u32) !*Self {
        const vkbd = try BaseDispatch.load(vkGetInstanceProcAddr);

        const app_info: vk.ApplicationInfo = .{
            .p_application_name = "vulkan_zig_video_encode",
            .application_version = @bitCast(vk.makeApiVersion(0, 0, 0, 0)),
            .p_engine_name = "vulkan_zig_video_encode",
            .engine_version = @bitCast(vk.makeApiVersion(0, 0, 0, 0)),
            .api_version = @bitCast(vk.API_VERSION_1_4),
        };

        var extension_names = std.ArrayList([*:0]const u8).init(allocator);
        defer extension_names.deinit();

        if (DEBUG) {
            try extension_names.append(vk.extensions.ext_debug_utils.name);
        }

        const validation_layers = [_][*:0]const u8{"VK_LAYER_KHRONOS_validation"};
        const enabled_layers: []const [*:0]const u8 = if (DEBUG) &validation_layers else &.{};

        const instance_def = try vkbd.createInstance(&.{
            .p_application_info = &app_info,

            .enabled_extension_count = @intCast(extension_names.items.len),
            .pp_enabled_extension_names = extension_names.items.ptr,

            .enabled_layer_count = @intCast(enabled_layers.len),
            .pp_enabled_layer_names = enabled_layers.ptr,
        }, null);

        const vki = try allocator.create(InstanceDispatch);
        errdefer allocator.destroy(vki);
        vki.* = try InstanceDispatch.load(instance_def, vkbd.dispatch.vkGetInstanceProcAddr);
        const instance = Instance.init(instance_def, vki);
        errdefer instance.destroyInstance(null);

        var debug_messenger: ?vk.DebugUtilsMessengerEXT = null;

        if (DEBUG) {
            debug_messenger = try instance.createDebugUtilsMessengerEXT(&.{
                .message_severity = .{
                    .error_bit_ext = true,
                    .warning_bit_ext = true,
                },
                .message_type = .{
                    .general_bit_ext = true,
                    .validation_bit_ext = true,
                    .performance_bit_ext = true,
                    .device_address_binding_bit_ext = true,
                },
                .pfn_user_callback = debugCallback,
            }, null);
        }
        errdefer {
            if (debug_messenger) |dm| {
                instance.destroyDebugUtilsMessengerEXT(dm, null);
            }
        }

        const candidate = try pickPhysicalDevice(instance, allocator);

        const pdev = candidate.pdev;
        const props = candidate.props;
        const mem_props = instance.getPhysicalDeviceMemoryProperties(pdev);

        const device_candidate = try initializeCandidate(instance, candidate);
        const vkd = try allocator.create(DeviceDispatch);
        errdefer allocator.destroy(vkd);
        vkd.* = try DeviceDispatch.load(device_candidate, instance.wrapper.dispatch.vkGetDeviceProcAddr);
        const device = Device.init(device_candidate, vkd);
        errdefer device.destroyDevice(null);

        const graphics_queue = Queue.init(device, candidate.queues.graphics_family);
        const video_encode_queue = Queue.init(device, candidate.queues.video_encode_family);

        // We use an allocator here because we don't want the
        // reference to change when we return this object.
        var self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .vkb = vkbd,
            .instance = instance,
            .debug_messenger = debug_messenger,
            .device = device,
            .graphics_queue = graphics_queue,
            .video_encode_queue = video_encode_queue,
            .physical_device = pdev,
            .props = props,
            .mem_props = mem_props,
            .height = height,
            .width = width,
            .images = undefined,
            .image_views = undefined,
            .image_memory = std.ArrayList(vk.DeviceMemory).init(allocator),
            .graphics_pipeline_layout = undefined,
            .graphics_pipeline = undefined,
            .command_pool = undefined,
            .command_buffers = std.ArrayList(vk.CommandBuffer).init(allocator),
            .video_encoder = undefined,
        };

        const dname: []const u8 = std.mem.sliceTo(&self.props.device_name, 0);
        std.debug.print("found device: {s}\n", .{dname});

        try self.initImages();
        errdefer self.destroyImages();

        try self.createCommandPool();
        errdefer device.destroyCommandPool(self.command_pool, null);

        try self.initGraphicsPipeline();
        errdefer {
            device.destroyPipeline(self.graphics_pipeline, null);
            device.destroyPipelineLayout(self.graphics_pipeline_layout, null);
        }

        try self.createCommandBuffers();
        self.video_encoder = try VideoEncoder.init(
            self.allocator,
            self,
            self.physical_device,
            self.device,
            self.instance,
            self.graphics_queue.family,
            self.graphics_queue.handle,
            self.command_pool,
            self.video_encode_queue.family,
            self.video_encode_queue.handle,
            self.images,
            self.image_views,
            self.width,
            self.height,
            FPS,
        );
        errdefer self.video_encoder.deinit();

        self.outfile = try std.fs.cwd().createFile("out.h264", .{ .read = true });

        return self;
    }

    pub fn mainLoop(self: *Self) !void {
        for (0..TOTAL_FRAMES) |i| {
            const current_frame_ix: u32 = @intCast(i % self.images.items.len);
            try self.drawFrame(current_frame_ix, @intCast(i));
            try self.encodeFrame(current_frame_ix);
        }
    }

    fn initImages(self: *Self) !void {
        self.images = std.ArrayList(vk.Image).init(self.allocator);

        // Create iamges
        for (0..IMAGE_COUNT) |_| {
            const image_create_info = vk.ImageCreateInfo{
                .image_type = .@"2d",
                .format = .r8g8b8a8_unorm,
                .extent = .{ .height = self.height, .width = self.width, .depth = 1 },
                .mip_levels = 1,
                .array_layers = 1,
                .samples = .{ .@"1_bit" = true },
                .tiling = .optimal,
                .usage = .{ .color_attachment_bit = true, .storage_bit = true },
                .sharing_mode = .exclusive,
                .initial_layout = .undefined,
            };
            const image = try self.device.createImage(&image_create_info, null);
            try self.images.append(image);
            const memory_reqs = self.device.getImageMemoryRequirements(image);

            const memory = try self.allocate(memory_reqs, .{});
            try self.image_memory.append(memory);
            try self.device.bindImageMemory(image, memory, 0);
        }

        self.image_views = std.ArrayList(vk.ImageView).init(self.allocator);

        // Create image views
        for (0..IMAGE_COUNT) |i| {
            const view_info = vk.ImageViewCreateInfo{
                .image = self.images.items[i],
                .view_type = .@"2d",
                .format = .r8g8b8a8_unorm,
                .subresource_range = .{
                    .aspect_mask = .{ .color_bit = true },
                    .base_mip_level = 0,
                    .level_count = 1,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
                .components = .{
                    .r = .r,
                    .g = .g,
                    .b = .b,
                    .a = .a,
                },
            };
            const image_view = try self.device.createImageView(&view_info, null);
            try self.image_views.append(image_view);
        }
    }

    fn initGraphicsPipeline(self: *Self) !void {
        const vert_spv = @embedFile("random_vert_shader");
        const vert = try self.device.createShaderModule(&.{
            .code_size = vert_spv.len,
            .p_code = @alignCast(@ptrCast(vert_spv)),
        }, null);
        defer self.device.destroyShaderModule(vert, null);

        const frag_spv = @embedFile("random_frag_shader");
        const frag = try self.device.createShaderModule(&.{
            .code_size = frag_spv.len,
            .p_code = @alignCast(@ptrCast(frag_spv)),
        }, null);
        defer self.device.destroyShaderModule(frag, null);

        const shader_stages = [_]vk.PipelineShaderStageCreateInfo{
            .{
                .stage = .{ .vertex_bit = true },
                .module = vert,
                .p_name = "main",
            },
            .{
                .stage = .{ .fragment_bit = true },
                .module = frag,
                .p_name = "main",
            },
        };

        const push_constant_range = [_]vk.PushConstantRange{.{
            .stage_flags = .{ .vertex_bit = true },
            .offset = 0,
            .size = @sizeOf(u32),
        }};

        self.graphics_pipeline_layout = try self.device.createPipelineLayout(&.{
            .push_constant_range_count = push_constant_range.len,
            .p_push_constant_ranges = &push_constant_range,
        }, null);

        const vertext_input_info = vk.PipelineVertexInputStateCreateInfo{};
        const input_assembly = vk.PipelineInputAssemblyStateCreateInfo{
            .topology = .triangle_list,
            .primitive_restart_enable = vk.FALSE,
        };
        const viewport = [_]vk.Viewport{.{
            .x = 0,
            .y = 0,
            .height = @floatFromInt(self.height),
            .width = @floatFromInt(self.width),
            .min_depth = 0,
            .max_depth = 1,
        }};

        const scissor = [_]vk.Rect2D{.{
            .offset = .{ .x = 0, .y = 0 },
            .extent = .{
                .height = self.height,
                .width = self.width,
            },
        }};

        const viewport_state = vk.PipelineViewportStateCreateInfo{
            .viewport_count = 1,
            .p_viewports = &viewport,
            .scissor_count = 1,
            .p_scissors = &scissor,
        };

        const rasterizer = vk.PipelineRasterizationStateCreateInfo{
            .depth_clamp_enable = vk.FALSE,
            .rasterizer_discard_enable = vk.FALSE,
            .polygon_mode = .fill,
            .cull_mode = .{ .back_bit = true },
            .front_face = .clockwise,
            .depth_bias_enable = vk.TRUE,
            .depth_bias_constant_factor = 0,
            .depth_bias_clamp = 0,
            .depth_bias_slope_factor = 0,
            .line_width = 1,
        };

        const multisampling = vk.PipelineMultisampleStateCreateInfo{
            .rasterization_samples = .{ .@"1_bit" = true },
            .sample_shading_enable = vk.FALSE,
            .min_sample_shading = 1,
            .p_sample_mask = null,
            .alpha_to_coverage_enable = vk.FALSE,
            .alpha_to_one_enable = vk.FALSE,
        };

        const color_blend_attachment = [_]vk.PipelineColorBlendAttachmentState{.{
            .blend_enable = vk.FALSE,
            .src_color_blend_factor = .one,
            .dst_color_blend_factor = .zero,
            .color_blend_op = .add,
            .src_alpha_blend_factor = .one,
            .dst_alpha_blend_factor = .zero,
            .alpha_blend_op = .add,
            .color_write_mask = .{
                .r_bit = true,
                .g_bit = true,
                .b_bit = true,
                .a_bit = true,
            },
        }};

        const color_blending = vk.PipelineColorBlendStateCreateInfo{
            .logic_op_enable = vk.FALSE,
            .logic_op = .copy,
            .attachment_count = 1,
            .p_attachments = &color_blend_attachment,
            .blend_constants = .{ 0, 0, 0, 0 },
        };

        const color_attachment_format = [_]vk.Format{.r8g8b8a8_unorm};

        const rendering_create_info = vk.PipelineRenderingCreateInfo{
            .view_mask = 0,
            .color_attachment_count = 1,
            .p_color_attachment_formats = &color_attachment_format,
            .depth_attachment_format = .undefined,
            .stencil_attachment_format = .undefined,
        };

        const pipeline_info = vk.GraphicsPipelineCreateInfo{
            .p_next = &rendering_create_info,
            .stage_count = 2,
            .p_stages = &shader_stages,
            .p_vertex_input_state = &vertext_input_info,
            .p_input_assembly_state = &input_assembly,
            .p_viewport_state = &viewport_state,
            .p_rasterization_state = &rasterizer,
            .p_multisample_state = &multisampling,
            .p_depth_stencil_state = null,
            .p_color_blend_state = &color_blending,
            .p_dynamic_state = null,
            .layout = self.graphics_pipeline_layout,
            .render_pass = .null_handle,
            .subpass = 0,
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = -1,
        };

        _ = try self.device.createGraphicsPipelines(
            .null_handle,
            1,
            @ptrCast(&pipeline_info),
            null,
            @ptrCast(&self.graphics_pipeline),
        );
    }

    fn createCommandPool(self: *Self) !void {
        self.command_pool = try self.device.createCommandPool(&.{
            .queue_family_index = self.graphics_queue.family,
            .flags = .{ .reset_command_buffer_bit = true },
        }, null);
    }

    fn createCommandBuffers(self: *Self) !void {
        try self.command_buffers.resize(self.images.items.len);
        const alloc_info = vk.CommandBufferAllocateInfo{
            .command_pool = self.command_pool,
            .level = .primary,
            .command_buffer_count = @intCast(self.command_buffers.items.len),
        };
        _ = try self.device.allocateCommandBuffers(&alloc_info, self.command_buffers.items.ptr);
    }

    fn pickPhysicalDevice(
        instance: Instance,
        allocator: std.mem.Allocator,
    ) !DeviceCandidate {
        const pdevs = try instance.enumeratePhysicalDevicesAlloc(allocator);
        defer allocator.free(pdevs);

        for (pdevs) |pdev| {
            if (try checkSuitable(instance, pdev, allocator)) |candidate| {
                return candidate;
            }
        }

        return error.NoSuitableDevice;
    }

    fn debugCallback(
        message_severity: vk.DebugUtilsMessageSeverityFlagsEXT,
        message_types: vk.DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT,
        p_user_data: ?*anyopaque,
    ) callconv(vk.vulkan_call_conv) vk.Bool32 {
        _ = message_severity;
        _ = message_types;
        _ = p_user_data;
        b: {
            const msg = (p_callback_data orelse break :b).p_message orelse break :b;
            std.log.scoped(.validation).warn("{s}", .{msg});
            return vk.FALSE;
        }
        std.log.scoped(.validation).warn("unrecognized validation layer debug message", .{});
        return vk.FALSE;
    }

    fn checkSuitable(
        instance: Instance,
        pdev: vk.PhysicalDevice,
        allocator: std.mem.Allocator,
    ) !?DeviceCandidate {
        if (!try checkExtensionSupport(instance, pdev, allocator)) {
            return null;
        }

        if (try allocateQueues(instance, pdev, allocator)) |allocation| {
            const props = instance.getPhysicalDeviceProperties(pdev);
            return DeviceCandidate{
                .pdev = pdev,
                .props = props,
                .queues = allocation,
            };
        }

        return null;
    }

    fn checkExtensionSupport(
        instance: Instance,
        pdev: vk.PhysicalDevice,
        allocator: std.mem.Allocator,
    ) !bool {
        const propsv = try instance.enumerateDeviceExtensionPropertiesAlloc(pdev, null, allocator);
        defer allocator.free(propsv);

        for (DEVICE_EXTENSIONS) |ext| {
            for (propsv) |props| {
                if (std.mem.eql(u8, std.mem.span(ext), std.mem.sliceTo(&props.extension_name, 0))) {
                    break;
                }
            } else {
                return false;
            }
        }

        return true;
    }

    fn allocateQueues(
        instance: Instance,
        pdev: vk.PhysicalDevice,
        allocator: std.mem.Allocator,
    ) !?QueueAllocation {
        const families = try instance.getPhysicalDeviceQueueFamilyPropertiesAlloc(pdev, allocator);
        defer allocator.free(families);

        var graphics_family: ?u32 = null;
        var video_encode_family: ?u32 = null;

        for (families, 0..) |properties, i| {
            const family: u32 = @intCast(i);

            if (graphics_family == null and properties.queue_flags.graphics_bit) {
                graphics_family = family;
            }

            if (video_encode_family == null and properties.queue_flags.video_encode_bit_khr) {
                video_encode_family = family;
            }
        }

        if (graphics_family != null and video_encode_family != null) {
            return QueueAllocation{
                .graphics_family = graphics_family.?,
                .video_encode_family = video_encode_family.?,
            };
        }

        return null;
    }

    /// - create device
    /// - add device extensions
    /// - add device queues
    fn initializeCandidate(instance: Instance, candidate: DeviceCandidate) !vk.Device {
        const priority = [_]f32{1};
        const qci = [_]vk.DeviceQueueCreateInfo{
            .{
                .queue_family_index = candidate.queues.graphics_family,
                .queue_count = 1,
                .p_queue_priorities = &priority,
            },
            .{
                .queue_family_index = candidate.queues.video_encode_family,
                .queue_count = 1,
                .p_queue_priorities = &priority,
            },
        };

        const queue_count: u32 = if (candidate.queues.graphics_family == candidate.queues.video_encode_family)
            1
        else
            2;

        const synchronization2_features = vk.PhysicalDeviceSynchronization2Features{
            .synchronization_2 = vk.TRUE,
        };

        const dynamic_rendering_features = vk.PhysicalDeviceDynamicRenderingFeaturesKHR{
            .p_next = @constCast(@ptrCast(&synchronization2_features)),
            .dynamic_rendering = vk.TRUE,
        };

        return try instance.createDevice(candidate.pdev, &.{
            .p_next = &dynamic_rendering_features,
            .queue_create_info_count = queue_count,
            .p_queue_create_infos = &qci,
            .enabled_extension_count = DEVICE_EXTENSIONS.len,
            .pp_enabled_extension_names = @ptrCast(&DEVICE_EXTENSIONS),
        }, null);
    }

    pub fn allocate(self: *Self, requirements: vk.MemoryRequirements, flags: vk.MemoryPropertyFlags) !vk.DeviceMemory {
        return try self.device.allocateMemory(&.{
            .allocation_size = requirements.size,
            .memory_type_index = try self.findMemoryTypeIndex(requirements.memory_type_bits, flags),
        }, null);
    }

    pub fn findMemoryTypeIndex(self: *Self, memory_type_bits: u32, flags: vk.MemoryPropertyFlags) !u32 {
        for (self.mem_props.memory_types[0..self.mem_props.memory_type_count], 0..) |mem_type, i| {
            if (memory_type_bits & (@as(u32, 1) << @truncate(i)) != 0 and mem_type.property_flags.contains(flags)) {
                return @truncate(i);
            }
        }

        return error.NoSuitableMemoryType;
    }

    pub fn drawFrame(self: *Self, current_image_ix: u32, current_frame_number: u32) !void {
        try self.device.resetCommandBuffer(
            self.command_buffers.items[current_image_ix],
            .{},
        );
        try self.recordCommandBuffer(
            self.command_buffers.items[current_image_ix],
            current_image_ix,
            current_frame_number,
        );

        const submit_info = vk.SubmitInfo{
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&self.command_buffers.items[current_image_ix]),
        };

        try self.device.queueSubmit(
            self.graphics_queue.handle,
            1,
            @ptrCast(&submit_info),
            .null_handle,
        );
    }

    pub fn encodeFrame(self: *Self, current_image_ix: u32) !void {

        // finish encoding the previous frame
        var packet_size: usize = 0;

        while (true) {
            var packet_data: ?[*]u8 = null;
            try self.video_encoder.finishEncode(&packet_data, &packet_size);
            if (packet_data) |pd| {
                try self.outfile.?.writeAll(pd[0..packet_size]);
            }
            if (packet_size <= 0) {
                break;
            }
        }

        try self.video_encoder.queueEncode(current_image_ix);
    }

    fn recordCommandBuffer(
        self: *Self,
        command_buffer: vk.CommandBuffer,
        current_image_ix: u32,
        current_frame_number: u32,
    ) !void {
        try self.device.beginCommandBuffer(command_buffer, &.{});

        const image_memory_barrier = vk.ImageMemoryBarrier2{
            .src_stage_mask = .{ .compute_shader_bit = true },
            .src_access_mask = .{ .shader_storage_read_bit = true },
            .dst_stage_mask = .{ .color_attachment_output_bit = true },
            .dst_access_mask = .{ .color_attachment_write_bit = true },
            .old_layout = .undefined,
            .new_layout = .attachment_optimal,
            .image = self.images.items[current_image_ix],
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

        self.device.cmdPipelineBarrier2(command_buffer, &dependency_info);

        const clear_value = vk.ClearValue{ .color = .{ .float_32 = [4]f32{
            0.0,
            0.0,
            0.0,
            0.0,
        } } };

        const color_attachment_info = vk.RenderingAttachmentInfo{
            .image_view = self.image_views.items[current_image_ix],
            .image_layout = .attachment_optimal,
            .load_op = .clear,
            .store_op = .store,
            .clear_value = clear_value,
            .resolve_mode = .{},
            .resolve_image_layout = .undefined,
        };

        const render_info = vk.RenderingInfo{
            .render_area = .{
                .extent = .{
                    .height = self.height,
                    .width = self.width,
                },
                .offset = .{
                    .x = 0,
                    .y = 0,
                },
            },
            .layer_count = 1,
            .color_attachment_count = 1,
            .p_color_attachments = @ptrCast(&color_attachment_info),
            .view_mask = 0,
        };

        self.device.cmdBeginRendering(command_buffer, &render_info);
        self.device.cmdBindPipeline(command_buffer, .graphics, self.graphics_pipeline);
        self.device.cmdPushConstants(
            command_buffer,
            self.graphics_pipeline_layout,
            .{ .vertex_bit = true },
            0,
            @sizeOf(u32),
            &current_frame_number,
        );
        self.device.cmdDraw(command_buffer, 3, 1, 0, 0);
        self.device.cmdEndRendering(command_buffer);
        try self.device.endCommandBuffer(command_buffer);
    }

    fn destroyImages(self: *Self) void {
        for (self.images.items) |image| {
            self.device.destroyImage(image, null);
        }
        for (self.image_views.items) |image_view| {
            self.device.destroyImageView(image_view, null);
        }
        for (self.image_memory.items) |memory| {
            self.device.freeMemory(memory, null);
        }
    }

    pub fn deinit(self: *VulkanApp) void {
        self.video_encoder.deinit();
        if (self.debug_messenger) |debug_messenger| {
            self.instance.destroyDebugUtilsMessengerEXT(debug_messenger, null);
        }
        self.destroyImages();

        self.outfile.?.close();
        self.device.destroyCommandPool(self.command_pool, null);
        self.device.destroyPipeline(self.graphics_pipeline, null);
        self.device.destroyPipelineLayout(self.graphics_pipeline_layout, null);

        self.device.destroyDevice(null);
        self.instance.destroyInstance(null);

        // allocator destroys
        self.allocator.destroy(self.device.wrapper);
        self.allocator.destroy(self.instance.wrapper);
        self.images.deinit();
        self.image_views.deinit();
        self.image_memory.deinit();
        self.command_buffers.deinit();
        self.allocator.destroy(self);
    }
};
