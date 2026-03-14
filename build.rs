use std::path::PathBuf;
use std::process::Command;
use std::{env, fs};

/// MNN linking mode
enum MnnLinkMode {
    /// Build MNN from source (default)
    BuildFromSource,
    /// Use pre-built MNN dynamic library
    Dynamic,
    /// Use pre-built MNN static library
    Static,
}

fn main() {
    // 在 docs.rs 构建环境中，跳过所有 C++ 编译
    if env::var("DOCS_RS").is_ok() || env::var("CARGO_FEATURE_DOCSRS").is_ok() {
        println!("cargo:warning=Building for docs.rs, skipping C++ compilation");
        return;
    }

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let debug = env::var("DEBUG").unwrap();

    // Feature flags
    let coreml_enabled = env::var("CARGO_FEATURE_COREML").is_ok();
    let metal_enabled = env::var("CARGO_FEATURE_METAL").is_ok();
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    let opencl_enabled = env::var("CARGO_FEATURE_OPENCL").is_ok();
    let opengl_enabled = env::var("CARGO_FEATURE_OPENGL").is_ok();
    let vulkan_enabled = env::var("CARGO_FEATURE_VULKAN").is_ok();

    let mnn_dynamic = env::var("CARGO_FEATURE_MNN_DYNAMIC").is_ok();
    let mnn_static = env::var("CARGO_FEATURE_MNN_STATIC").is_ok();

    if mnn_dynamic && mnn_static {
        panic!("Features `mnn-dynamic` and `mnn-static` are mutually exclusive. Please enable only one.");
    }

    let link_mode = if mnn_dynamic {
        MnnLinkMode::Dynamic
    } else if mnn_static {
        MnnLinkMode::Static
    } else {
        MnnLinkMode::BuildFromSource
    };

    let manifest_dir_path = PathBuf::from(&manifest_dir);

    // Determine MNN include dir and library dir based on link mode
    let (mnn_include_dir, mnn_lib_dir) = match &link_mode {
        MnnLinkMode::BuildFromSource => {
            // Get or download MNN source code
            let mnn_source_dir = get_mnn_source(&manifest_dir_path);

            // Build MNN using cmake
            let dst = build_mnn_with_cmake(
                &mnn_source_dir,
                &arch,
                &os,
                &debug,
                coreml_enabled,
                metal_enabled,
                cuda_enabled,
                opencl_enabled,
                opengl_enabled,
                vulkan_enabled,
            );

            // Include dirs: cmake output + MNN source
            let include_dir = vec![dst.join("include"), mnn_source_dir.join("include")];
            let lib_dir = vec![dst.clone(), dst.join("lib")];
            (include_dir, lib_dir)
        }
        MnnLinkMode::Dynamic | MnnLinkMode::Static => {
            let mode_name = if mnn_dynamic {
                "mnn-dynamic"
            } else {
                "mnn-static"
            };

            // MNN_LIB_DIR is required for pre-built libraries
            let lib_dir_str = env::var("MNN_LIB_DIR").unwrap_or_else(|_| {
                panic!(
                    "MNN_LIB_DIR environment variable is required when using `{}` feature.\n\
                     Set it to the directory containing the pre-built MNN library.\n\
                     Example: MNN_LIB_DIR=/usr/local/lib cargo build --features {}",
                    mode_name, mode_name,
                )
            });
            let lib_dir = PathBuf::from(&lib_dir_str);
            if !lib_dir.exists() {
                panic!("MNN_LIB_DIR='{}' does not exist", lib_dir.display());
            }

            // MNN_INCLUDE_DIR: look for it in env, or fall back to MNN source/3rd_party
            let include_dirs = get_mnn_include_dirs(&manifest_dir_path);

            println!("cargo:rerun-if-env-changed=MNN_LIB_DIR");
            println!("cargo:rerun-if-env-changed=MNN_INCLUDE_DIR");

            println!(
                "cargo:warning=Using pre-built MNN {} library from: {}",
                if mnn_dynamic { "dynamic" } else { "static" },
                lib_dir.display()
            );

            (include_dirs, vec![lib_dir])
        }
    };

    // Build our C++ wrapper using cc (always needed)
    build_wrapper(&manifest_dir_path, &mnn_include_dir, &os);

    // Link libraries
    link_libraries(
        &mnn_lib_dir,
        &os,
        &link_mode,
        coreml_enabled,
        metal_enabled,
        cuda_enabled,
        opencl_enabled,
        opengl_enabled,
        vulkan_enabled,
    );

    // Generate Rust bindings
    bind_gen(&manifest_dir_path, &mnn_include_dir, &os, &arch);
}

/// Get MNN include directories for pre-built library mode.
/// Priority:
/// 1. MNN_INCLUDE_DIR environment variable
/// 2. MNN_SOURCE_DIR/include (if MNN_SOURCE_DIR is set)
/// 3. Local 3rd_party/MNN/include
fn get_mnn_include_dirs(manifest_dir: &PathBuf) -> Vec<PathBuf> {
    // 1. Check MNN_INCLUDE_DIR
    if let Ok(include_dir) = env::var("MNN_INCLUDE_DIR") {
        let include_path = PathBuf::from(&include_dir);
        if include_path.exists() {
            println!(
                "cargo:warning=Using MNN headers from MNN_INCLUDE_DIR: {}",
                include_path.display()
            );
            return vec![include_path];
        } else {
            panic!(
                "MNN_INCLUDE_DIR='{}' does not exist",
                include_path.display()
            );
        }
    }

    // 2. Check MNN_SOURCE_DIR
    if let Ok(mnn_dir) = env::var("MNN_SOURCE_DIR") {
        let mnn_path = PathBuf::from(&mnn_dir);
        let include_path = mnn_path.join("include");
        if include_path.exists() {
            println!(
                "cargo:warning=Using MNN headers from MNN_SOURCE_DIR: {}",
                include_path.display()
            );
            return vec![include_path];
        }
    }

    // 3. Check local 3rd_party/MNN/include
    let local_include = manifest_dir.join("3rd_party/MNN/include");
    if local_include.exists() {
        println!(
            "cargo:warning=Using MNN headers from local source: {}",
            local_include.display()
        );
        return vec![local_include];
    }

    panic!(
        "MNN headers not found. Please set one of:\n\
         - MNN_INCLUDE_DIR: path to directory containing MNN headers\n\
         - MNN_SOURCE_DIR: path to MNN source tree\n\
         Or ensure 3rd_party/MNN exists in the project root."
    );
}

/// Get MNN source code directory
/// Priority:
/// 1. Environment variable MNN_SOURCE_DIR
/// 2. Local 3rd_party/MNN directory
/// 3. Clone from GitHub
fn get_mnn_source(manifest_dir: &PathBuf) -> PathBuf {
    // Check environment variable first
    if let Ok(mnn_dir) = env::var("MNN_SOURCE_DIR") {
        let mnn_path = PathBuf::from(mnn_dir);
        if mnn_path.exists() && mnn_path.join("CMakeLists.txt").exists() {
            println!(
                "cargo:warning=Using MNN source from MNN_SOURCE_DIR: {}",
                mnn_path.display()
            );
            return mnn_path;
        } else {
            panic!(
                "MNN_SOURCE_DIR is set but directory is invalid or missing CMakeLists.txt: {}",
                mnn_path.display()
            );
        }
    }

    // Check local 3rd_party/MNN
    let local_mnn = manifest_dir.join("3rd_party/MNN");
    if local_mnn.exists() && local_mnn.join("CMakeLists.txt").exists() {
        println!(
            "cargo:warning=Using local MNN source: {}",
            local_mnn.display()
        );
        return local_mnn;
    }

    // Clone from GitHub
    println!("cargo:warning=MNN source not found, cloning from GitHub...");
    let third_party_dir = manifest_dir.join("3rd_party");
    fs::create_dir_all(&third_party_dir).expect("Failed to create 3rd_party directory");

    let status = Command::new("git")
        .args(&[
            "clone",
            "--depth=1",
            "--branch=3.4.1",
            "https://github.com/alibaba/MNN.git",
            local_mnn.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute git clone command. Make sure git is installed.");

    if !status.success() {
        panic!("Failed to clone MNN from GitHub");
    }

    if !local_mnn.join("CMakeLists.txt").exists() {
        panic!("MNN cloned but CMakeLists.txt not found");
    }

    println!(
        "cargo:warning=Successfully cloned MNN to: {}",
        local_mnn.display()
    );
    local_mnn
}

fn build_mnn_with_cmake(
    mnn_source_dir: &PathBuf,
    arch: &str,
    os: &str,
    debug: &str,
    coreml_enabled: bool,
    metal_enabled: bool,
    cuda_enabled: bool,
    opencl_enabled: bool,
    opengl_enabled: bool,
    vulkan_enabled: bool,
) -> PathBuf {
    let mut config = cmake::Config::new(mnn_source_dir);

    config
        .define("MNN_BUILD_SHARED_LIBS", "OFF")
        .define("MNN_BUILD_TOOLS", "OFF")
        .define("MNN_BUILD_DEMO", "OFF")
        .define("MNN_BUILD_TEST", "OFF")
        .define("MNN_BUILD_BENCHMARK", "OFF")
        .define("MNN_BUILD_QUANTOOLS", "OFF")
        .define("MNN_BUILD_CONVERTER", "OFF")
        .define("MNN_PORTABLE_BUILD", "ON")
        .define("MNN_SEP_BUILD", "OFF");

    // For Windows, always use Release mode to ensure consistent CRT linking
    if os == "windows" {
        // Force NMake Makefiles generator on Windows to avoid MSVC detection issues
        // This is more reliable in CI/CD environments like Jenkins
        config.generator("NMake Makefiles");
        config.define("CMAKE_BUILD_TYPE", "Release");
        // Check if we're using static CRT
        if env::var("CARGO_CFG_TARGET_FEATURE").map_or(false, |f| f.contains("crt-static")) {
            // MNN has a specific option for static CRT on Windows
            config.define("MNN_WIN_RUNTIME_MT", "ON");

            // Also set these for extra safety
            config.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded");
            config.define("CMAKE_C_FLAGS_RELEASE", "/MT /O2 /Ob2 /DNDEBUG");
            config.define("CMAKE_CXX_FLAGS_RELEASE", "/MT /O2 /Ob2 /DNDEBUG");
            config.define("CMAKE_C_FLAGS", "/MT");
            config.define("CMAKE_CXX_FLAGS", "/MT");
        }
    } else {
        // For non-Windows platforms, respect debug flag
        if debug == "true" {
            config.define("CMAKE_BUILD_TYPE", "Debug");
        } else {
            config.define("CMAKE_BUILD_TYPE", "Release");
        }
    }

    // Android cross-compilation
    if os == "android" {
        let ndk = env::var("ANDROID_NDK_ROOT")
            .or_else(|_| env::var("ANDROID_NDK_HOME"))
            .or_else(|_| env::var("ANDROID_NDK"))
            .or_else(|_| env::var("NDK_HOME"))
            .expect(
                "Android NDK not found. Please set one of: ANDROID_NDK_ROOT, ANDROID_NDK_HOME, ANDROID_NDK, NDK_HOME",
            );

        config
            .define(
                "CMAKE_TOOLCHAIN_FILE",
                PathBuf::from(&ndk).join("build/cmake/android.toolchain.cmake"),
            )
            .define("ANDROID_STL", "c++_static")
            .define("ANDROID_NATIVE_API_LEVEL", "android-21")
            .define("ANDROID_TOOLCHAIN", "clang")
            .define("MNN_BUILD_FOR_ANDROID_COMMAND", "ON")
            .define("MNN_USE_SSE", "OFF");

        match arch {
            "arm" => {
                config.define("ANDROID_ABI", "armeabi-v7a");
            }
            "aarch64" => {
                config.define("ANDROID_ABI", "arm64-v8a");
            }
            "x86" => {
                config.define("ANDROID_ABI", "x86");
            }
            "x86_64" => {
                config.define("ANDROID_ABI", "x86_64");
            }
            _ => {}
        }
    }

    // iOS cross-compilation
    if os == "ios" {
        let rust_target = env::var("TARGET").unwrap_or_default();
        let is_simulator = rust_target.contains("-sim") || arch == "x86_64";

        config
            .define("CMAKE_SYSTEM_NAME", "iOS")
            .define("MNN_BUILD_FOR_IOS", "ON")
            .define("CMAKE_OSX_DEPLOYMENT_TARGET", "13.0");

        if arch == "aarch64" {
            config.define("CMAKE_OSX_ARCHITECTURES", "arm64");
        } else if arch == "x86_64" {
            config.define("CMAKE_OSX_ARCHITECTURES", "x86_64");
        }

        // Critical: set the correct SDK for simulator vs device
        if is_simulator {
            config.define("CMAKE_OSX_SYSROOT", "iphonesimulator");
        } else {
            config.define("CMAKE_OSX_SYSROOT", "iphoneos");
        }

        // MNN's CMakeLists.txt only sets CMAKE_SYSTEM_PROCESSOR from
        // CMAKE_OSX_ARCHITECTURES when CMAKE_SYSTEM_NAME == "Darwin",
        // but for iOS it's "iOS". Without this, ARM assembly sources
        // (NEON, AArch64) are not compiled, causing undefined symbols.
        if arch == "aarch64" {
            config.define("CMAKE_SYSTEM_PROCESSOR", "arm64");
            config.define("ARCHS", "arm64");
        } else if arch == "x86_64" {
            config.define("CMAKE_SYSTEM_PROCESSOR", "x86_64");
        }
    }

    // SIMD optimizations
    // Only enable SSE for x86_64, not for 32-bit x86 (i686)
    // because i686 target doesn't have guaranteed SSE support
    if arch == "x86_64" && os != "android" && os != "ios" {
        config.define("MNN_USE_SSE", "ON");
    } else {
        // For all other architectures (including 32-bit x86/i686), disable SSE/AVX
        // This prevents compilation errors with SIMD intrinsics on incompatible targets
        config.define("MNN_USE_SSE", "OFF");
        config.define("MNN_USE_AVX", "OFF");
        config.define("MNN_USE_AVX2", "OFF");
        config.define("MNN_USE_AVX512", "OFF");
    }

    // CoreML (macOS/iOS only)
    if coreml_enabled && matches!(os, "macos" | "ios") {
        config.define("MNN_COREML", "ON");
    }

    // Metal GPU (macOS/iOS only)
    if metal_enabled && matches!(os, "macos" | "ios") {
        config.define("MNN_METAL", "ON");
    }

    // CUDA GPU (Linux/Windows)
    if cuda_enabled && matches!(os, "linux" | "windows") {
        config.define("MNN_CUDA", "ON");
    }

    // OpenCL GPU (cross-platform)
    if opencl_enabled {
        config.define("MNN_OPENCL", "ON");
    }

    // OpenGL GPU (Android/Linux)
    if opengl_enabled && matches!(os, "android" | "linux") {
        config.define("MNN_OPENGL", "ON");
    }

    // Vulkan GPU (cross-platform)
    if vulkan_enabled {
        config.define("MNN_VULKAN", "ON");
    }

    println!("cargo:rerun-if-changed=MNN/CMakeLists.txt");

    config.build()
}

fn build_wrapper(manifest_dir: &PathBuf, mnn_include_dirs: &[PathBuf], os: &str) {
    let wrapper_file = manifest_dir.join("cpp/src/mnn_wrapper.cpp");

    println!("cargo:rerun-if-changed=cpp/src/mnn_wrapper.cpp");
    println!("cargo:rerun-if-changed=cpp/include/mnn_wrapper.h");

    let mut build = cc::Build::new();

    build
        .cpp(true)
        .file(&wrapper_file)
        .include(manifest_dir.join("cpp/include"));

    for inc in mnn_include_dirs {
        build.include(inc);
    }

    // Platform-specific C++ flags
    if os == "windows" {
        build.flag("/std:c++14").flag("/EHsc").flag("/W3");
    } else {
        build.flag("-std=c++14").flag("-fvisibility=hidden");
    }

    build.compile("mnn_wrapper");
}

fn link_libraries(
    lib_dirs: &[PathBuf],
    os: &str,
    link_mode: &MnnLinkMode,
    coreml_enabled: bool,
    metal_enabled: bool,
    cuda_enabled: bool,
    opencl_enabled: bool,
    opengl_enabled: bool,
    vulkan_enabled: bool,
) {
    // Add library search paths
    for dir in lib_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }

    // Link MNN library based on mode
    match link_mode {
        MnnLinkMode::Dynamic => {
            println!("cargo:rustc-link-lib=dylib=MNN");
        }
        MnnLinkMode::Static | MnnLinkMode::BuildFromSource => {
            println!("cargo:rustc-link-lib=static=MNN");
        }
    }

    // Platform-specific C++ runtime
    match os {
        "macos" | "ios" => {
            println!("cargo:rustc-link-lib=c++");
        }
        "linux" => {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=pthread");
        }
        "android" => {
            println!("cargo:rustc-link-lib=c++_static");
            println!("cargo:rustc-link-lib=log");
        }
        "windows" => {
            // MSVC runtime is linked automatically when using matching CRT settings
        }
        _ => {}
    }

    // CoreML frameworks
    if coreml_enabled && matches!(os, "macos" | "ios") {
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    }

    // Metal frameworks
    if metal_enabled && matches!(os, "macos" | "ios") {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    }

    // CUDA libraries
    if cuda_enabled && matches!(os, "linux" | "windows") {
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudnn");
    }

    // OpenCL library
    if opencl_enabled {
        if os == "macos" {
            println!("cargo:rustc-link-lib=framework=OpenCL");
        } else {
            println!("cargo:rustc-link-lib=OpenCL");
        }
    }

    // OpenGL libraries
    if opengl_enabled && matches!(os, "android" | "linux") {
        if os == "android" {
            println!("cargo:rustc-link-lib=GLESv3");
            println!("cargo:rustc-link-lib=EGL");
        } else {
            println!("cargo:rustc-link-lib=GL");
        }
    }

    // Vulkan library
    if vulkan_enabled {
        println!("cargo:rustc-link-lib=vulkan");
    }
}

fn bind_gen(manifest_dir: &PathBuf, mnn_include_dirs: &[PathBuf], os: &str, arch: &str) {
    let header_path = manifest_dir.join("cpp/include/mnn_wrapper.h");

    let mut builder = bindgen::Builder::default()
        .header(header_path.to_string_lossy())
        .allowlist_function("mnnr_.*")
        .allowlist_type("MNN.*")
        .allowlist_type("MNNR.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .layout_tests(false);

    for inc in mnn_include_dirs {
        builder = builder.clang_arg(format!("-I{}", inc.display()));
    }

    // Android-specific clang target and sysroot
    if os == "android" {
        let ndk = env::var("ANDROID_NDK_ROOT")
            .or_else(|_| env::var("ANDROID_NDK_HOME"))
            .or_else(|_| env::var("ANDROID_NDK"))
            .or_else(|_| env::var("NDK_HOME"))
            .unwrap_or_default();

        let api_level = "21";
        let target = match arch {
            "aarch64" => "aarch64-linux-android",
            "arm" => "armv7-linux-androideabi",
            "x86_64" => "x86_64-linux-android",
            "x86" => "i686-linux-android",
            _ => "aarch64-linux-android",
        };
        builder = builder.clang_arg(format!("--target={}{}", target, api_level));

        // Point bindgen to the NDK sysroot so it doesn't pick up host headers
        if !ndk.is_empty() {
            let host_tag = if cfg!(target_os = "macos") {
                "darwin-x86_64"
            } else {
                "linux-x86_64"
            };
            let sysroot = PathBuf::from(&ndk)
                .join("toolchains/llvm/prebuilt")
                .join(host_tag)
                .join("sysroot");
            if sysroot.exists() {
                builder = builder.clang_arg(format!("--sysroot={}", sysroot.display()));
            }
        }
    }

    // iOS-specific: remap target triple for clang/bindgen compatibility
    if os == "ios" {
        let rust_target = env::var("TARGET").unwrap_or_default();
        let clang_target = if rust_target == "aarch64-apple-ios-sim" {
            "arm64-apple-ios13.0-simulator".to_string()
        } else if rust_target == "aarch64-apple-ios" {
            "arm64-apple-ios13.0".to_string()
        } else if rust_target == "x86_64-apple-ios" {
            "x86_64-apple-ios13.0-simulator".to_string()
        } else {
            rust_target
        };
        builder = builder.clang_arg(format!("--target={}", clang_target));
    }

    let bindings = builder.generate().expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::write(out_path.join("mnn_bindings.rs"), bindings.to_string())
        .expect("Couldn't write bindings!");
}
