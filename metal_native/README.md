# metal_native (Xu FFI)

Xu front-end for the Apple Metal bridge. The ObjC implementation stays in
`metal/native/bridge.m` (Metal/MPS cannot be expressed as pure Xu). Build it
as `libxu_metal.dylib` via `python3 metal/native/build.py`, then this package
`dlopen`s the dylib (override with `XU_METAL_LIBRARY`).

`xu.xun` no longer packs `metal.xulib` for `metal_native`; import
`github.com/xulang-org/packages/metal_native` instead of the old native map
entry.
