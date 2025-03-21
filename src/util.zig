const std = @import("std");

pub fn isWindows() bool {
    return @import("builtin").os.tag == .windows;
}

pub fn printElapsed(start_time: i128) void {
    const end = std.time.nanoTimestamp();
    const total_time = @divFloor(end - start_time, @as(i128, @intCast(std.time.ns_per_s)));
    std.debug.print("Time elapsed {} seconds\n", .{total_time});
}
