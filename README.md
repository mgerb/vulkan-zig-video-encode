# Vulkan Zig Video Encode

<div align="center">
  <img src="./output.gif" alt="h264 output video">
</div>

This is essentially a rewrite of https://github.com/clemy/vulkan-video-encode-simple using the following:

- [Zig](https://ziglang.org/)
- [vulkan-zig](https://github.com/Snektron/vulkan-zig)

It leverages the Zig build system to cross compile to Windows and produces the following binaries:

```
vulkan_zig_video_encode
vulkan_zig_video_encode.exe
```

## How it works

- generates a sample video stream
- converts RGB images to YCbCr using the compute pipeline
- uses Vulkan Video to encode images into `out.h264`

## Compiling

I've on compiled this on Linux, but it could probably be done on Windows without too much effort.
There are a few Vulkan validation warnings that need to be cleaned up, but it is still working.

### NixOS

```sh
nix-shell
zig build

# NixOS can't run dynamically linked executable so it needs some help
# See `shell.nix` for $PATCHELF
sh -c "$PATCHELF ./zig-out/bin/vulkan_zig_video_encode"

# execute
./zig-out/bin/vulkan_zig_video_encode
```

### Linux (using [Nix](https://nixos.org/))

```sh
# open `shell.nix` and comment out the line starting with LD_LIBRARY_PATH
nix-shell
zig build

# execute
./zig-out/bin/vulkan_zig_video_encode
```
