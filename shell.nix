with (import <nixpkgs> {});
  mkShell {
    buildInputs = [
      shaderc
      glfw
      vulkan-loader
      vulkan-validation-layers

      # windows
      pkgsCross.mingwW64.vulkan-loader

      glibc
    ];

    shellHook = ''
      exec zsh
    '';

    # NOTE: Execute with `sh -c "$PATCHELF ./zig-out/bin/vulkan_zig_video_encode" && ./zig-out/bin/vulkan_zig_video_encode`.
    # This is only required on NixOS because it cannot use global dynamically linked libraries.
    PATCHELF = "patchelf --set-interpreter ${glibc.out}/lib/ld-linux-x86-64.so.2";
    # NOTE: Uncomment this on any other OS than NixOS.
    # This causes a segfault with nix on anything other than NixOS.
    LD_LIBRARY_PATH = "${glibc.out}/lib:${pkgs.vulkan-loader}/lib:$LD_LIBRARY_PATH";

    VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
    VULKAN_SDK_PATH = "${pkgs.vulkan-loader}/lib";
    VULKAN_SDK_PATH_WINDOWS = "${pkgsCross.mingwW64.vulkan-loader}/bin";
  }
