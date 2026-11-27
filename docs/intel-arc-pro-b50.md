# Intel Arc Pro B50 on Linux

The Arc Pro B50 (16 GB, Xe2) is supported experimentally by inferzero's discrete Arc path. Linux is a practical choice for a headless server; start with the general [Linux installation guide](install.md#linux). This page adds B50-specific notes, not a separate installation path.

## One B50

Install the Intel OpenCL and Level Zero runtime required by the release. On Ubuntu 24.04, the distribution runtime may be too old for Battlemage; use the Intel graphics packages described in [Linux installation](install.md#linux) and [the Linux port notes](linux-port.md). The engine compiles kernels through OpenCL and runs them through Level Zero. The documented package also requires access to the GPU's render device; add the user to the `render` group and start a new login session.

Find the card's render node rather than assuming it is `renderD128`:

```sh
ls -l /dev/dri/by-path/*-render
```

Use the symlink whose PCI address matches the B50, for example `/dev/dri/by-path/pci-0000:03:00.0-render`. Render-node numbers can change when devices or drivers change. For a container, follow the project's [Docker files](../docker/) and expose `/dev/dri`; choose the intended card by its PCI path when the runtime or deployment supports selecting individual devices. Do not assume a fixed `renderD*` number identifies a particular GPU.

## Two B50s

There are two different setups:

- **One model per card:** run two independent `serverz` instances, each bound to its own GPU. This preserves each model's single-card execution path and lets the two servers serve separate models or workloads.
- **One model split across both cards:** use `serverz --gpus 2` (or `runz --gpus 2`) and see [Multi-GPU](multi-gpu.md) for split modes and options. This can provide capacity for a model that does not fit on one card, but is not automatically faster. The cards' PCIe topology, model architecture, release, and peer-to-peer (P2P) access affect performance.

Inferzero checks links at startup and selects direct GPU transfers only when the driver reports usable peer access; otherwise sums use host memory. P2P is optional, not a B50 requirement. Linux peer access depends on the driver, topology, and kernel support; the project's [Multi-GPU notes](multi-gpu.md) discuss `CONFIG_PCI_P2PDMA` and how to check it. Do not enable `ARC_TP_P2P=1` merely to force a faster path: it can request direct transfers when links are reported, but cannot create hardware or driver support. Leave the default automatic choice unless diagnosing a specific system.

Split-model capabilities differ from single-GPU models and can change by release and backend. The current documentation says image encoders and MTP / DFlash2 drafting are off for a split model; split-model vision and audio encoder support is planned but not available in the documented build (see [issue #103](https://github.com/rizagp/inferzero/issues/103)). Check the model's reported capabilities and documentation for the exact release before relying on vision, audio, or drafting; support for a model on one GPU does not imply support when split. See [issue #59](https://github.com/rizagp/inferzero/issues/59) for one dual-B50 Linux system with no peer access and its model-specific measurements. Those results are not a general speed guarantee.

For measured single- and dual-B50 results, including the tested release and setup, see the B50 entries in [results](../site/results.html) and the reports linked from [issue #59](https://github.com/rizagp/inferzero/issues/59). Performance varies with model, quantization, context, runtime, topology, and release. One separate single-B50 image test used Xiaomi-OCR-0 with a Q8_0 model and F16 projector on Inferzero `2026.11.26.03-6eaead5`, 8,192 context, temperature 0, and one labeled JPEG receipt: it took 46.17 seconds end to end (44.26 seconds to first token). That is one example, not an accuracy score or general OCR throughput benchmark; the logged token rates exclude the long first-token delay. See [issue #103](https://github.com/rizagp/inferzero/issues/103) for the discussion and split-encoder status.

## Unraid

The community-maintained [Inferzero Community Applications template and setup notes](https://github.com/catduckgnaf/catduck-unraid-templates/blob/main/README.md#inferzero) use the downstream image `ghcr.io/catduckgnaf/inferzero:latest`; that image and template are maintained separately from upstream Inferzero. Standard single-GPU inference does not require a custom kernel. For compatible multi-GPU systems, a separate community-maintained [Unraid P2P kernel project](https://github.com/catduckgnaf/unraid-p2p-kernel) documents its [version compatibility](https://github.com/catduckgnaf/unraid-p2p-kernel/blob/main/plugin/compatibility.tsv) and [manual installation and rollback](https://github.com/catduckgnaf/unraid-p2p-kernel/blob/main/INSTALL.md). It is optional and experimental, targets specific Unraid/kernel releases, and is not a promise of P2P support for every release or PCIe topology. Review its compatibility and safety notes before use; the Docker template never installs or modifies the host kernel.

## Docker and distribution packaging

The upstream [Docker Compose example](../docker/compose.yaml) passes `/dev/dri` to the container. Its Dockerfile installs Intel's OpenCL and Level Zero runtime packages. These are upstream examples; distribution-specific templates, image builds, and device mappings are maintained separately by their packagers. Check the current [Docker documentation](../docker/) and your distribution's device assignment before adapting them.
