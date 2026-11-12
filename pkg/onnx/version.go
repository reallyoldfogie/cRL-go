package onnx

// IRVersion is the ONNX IR version every model this module produces
// declares itself as targeting: onnx.proto's Version_IR_VERSION constant
// for the vendored schema (pinned to onnx/onnx tag v1.23.2 — see
// README.md's "ONNX support" section).
const IRVersion = int64(Version_IR_VERSION)

// OpsetVersion is the ai.onnx (default-domain) opset every model this
// module produces imports. 21, rather than the vendored schema's own
// ceiling (28, per onnx.defs.onnx_opset_version() in the matching Python
// onnx package), because every op this project needs (Gemm, Relu,
// Softmax, Sigmoid, Tanh, Add, Mul) has had stable semantics since long
// before opset 21 — staying well under the ceiling maximizes
// compatibility with whatever reads the resulting files (Netron,
// onnxruntime, anything else). See README.md for how to bump this
// alongside the vendored schema.
const OpsetVersion = 21
