const util = @import("util.zig");
const std = @import("std");
const vk = @import("vulkan");
const VulkanApp = @import("./vulkan_app.zig").VulkanApp;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const start_time = std.time.nanoTimestamp();
    const vulkan_app = try VulkanApp.init(allocator, 800, 600);
    defer vulkan_app.deinit();

    try vulkan_app.mainLoop();

    util.printElapsed(start_time);
}

test "simple test" {
    std.debug.print("test working...", .{});
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}
