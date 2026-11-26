# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-11-26

- fix(import): show the painted side in Solid view when the other side is unpainted (#25)
- feat(import): report partial classic .skp reads ahead of other warnings (#30)
- ci: test on Blender 4.2.4, 4.5.16 and 5.3.0, installing the release zip (#31)
- fix(import): keep one orientation across a painted surface (#34)
- feat(import): import SketchUp scenes as Blender cameras (#32)
- feat(export): export collection instances as SketchUp components (#36)
- feat(import): import SketchUp texts and dimensions (#37)
- feat(import,export): option to import components as linked duplicates and round-trip them as SketchUp components (#39)
- chore: bundle openskp 1.7.0, enabling scene cameras, 2021+ component annotations, dimension offsets with witness lines, style front/back unpainted face colours (#16), classic instance tags/paint, and canonical component GUIDs
- fix(import): add tag-colour option for unpainted faces, weld loose-edge curve vertices, keep stacked coplanar faces valid, preserve front paint on flat multi-material surfaces, and ignore all-opaque 32-bit texture alpha (#16)
