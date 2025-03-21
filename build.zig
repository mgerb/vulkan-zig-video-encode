const std = @import("std");

fn compileShader(
    allocator: std.mem.Allocator,
    b: *std.Build,
    exe: *std.Build.Step.Compile,
    shader: []const u8,
    importName: []const u8,
) !void {
    const vert_cmd = b.addSystemCommand(&.{
        "glslc",
        "--target-env=vulkan1.3",
        "-o",
    });
    const shaderPath = try std.fs.path.join(allocator, &[_][]const u8{ "libs", "shaders", shader });
    defer allocator.free(shaderPath);

    const outputFile = vert_cmd.addOutputFileArg(shader);
    vert_cmd.addFileArg(b.path(shaderPath));

    exe.root_module.addAnonymousImport(importName, .{
        .root_source_file = outputFile,
    });
}

fn add_shared_dependencies(
    allocator: std.mem.Allocator,
    b: *std.Build,
    exe: *std.Build.Step.Compile,
) !void {
    try compileShader(allocator, b, exe, "random.frag", "random_frag_shader");
    try compileShader(allocator, b, exe, "random.vert", "random_vert_shader");
    try compileShader(allocator, b, exe, "rgb-ycbcr-shader-2plane.comp", "rgb-ycbcr-shader-2plane");
    try compileShader(allocator, b, exe, "rgb-ycbcr-shader-3plane.comp", "rgb-ycbcr-shader-3plane");

    const vulkan_headers = b.dependency("vulkan_headers", .{});
    const vulkan = b.dependency(
        "vulkan_zig",
        .{
            .registry = vulkan_headers.path("registry/vk.xml"),
            .video = vulkan_headers.path("registry/video.xml"),
        },
    ).module("vulkan-zig");
    exe.root_module.addImport("vulkan", vulkan);

    exe.addLibraryPath(.{ .cwd_relative = std.posix.getenv("VULKAN_SDK_PATH_WINDOWS").? });
    exe.addLibraryPath(.{ .cwd_relative = std.posix.getenv("VULKAN_SDK_PATH").? });
}

fn buildWindows(
    allocator: std.mem.Allocator,
    b: *std.Build,
    optimize: std.builtin.OptimizeMode,
) !void {
    const windows_target = b.resolveTargetQuery(.{
        .os_tag = .windows,
        .abi = .gnu,
        .cpu_arch = .x86_64,
    });

    const exe = b.addExecutable(.{
        .name = "vulkan_zig_video_encode",
        .root_source_file = b.path("src/main.zig"),
        .target = windows_target,
        .optimize = optimize,
        .link_libc = true,
    });

    try add_shared_dependencies(allocator, b, exe);

    exe.linkSystemLibrary("vulkan-1");

    exe.linkSystemLibrary("gdi32");

    const zigwin32 = b.dependency("zigwin32", .{});
    exe.root_module.addImport("win32", zigwin32.module("win32"));

    b.installArtifact(exe);
}

fn buildLinux(
    allocator: std.mem.Allocator,
    b: *std.Build,
    optimize: std.builtin.OptimizeMode,
) !void {
    const exe = b.addExecutable(.{
        .name = "vulkan_zig_video_encode",
        .root_source_file = b.path("src/main.zig"),
        .target = b.resolveTargetQuery(.{
            .os_tag = .linux,
            .cpu_arch = .x86_64,
            .abi = .gnu,
        }),
        .optimize = optimize,
        .link_libc = true,
    });

    try add_shared_dependencies(allocator, b, exe);

    exe.linkSystemLibrary("vulkan");

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}

pub fn build(b: *std.Build) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    try buildLinux(allocator, b, optimize);
    try buildWindows(allocator, b, optimize);

    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step to
    // the `zig build --help` menu, providing a way for the user to request
    // running the unit tests.
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
}
