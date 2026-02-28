# bepe_pose

This repository is part of a **FAPESP-funded research project** (grant **2025/05613-8**) and consolidates code and resources developed for experimental optical sensing and pose-estimation workflows.

## Institutional Context

- Universidade Federal de São Carlos (UFSCar)
- Universidade de São Paulo (USP)
- Mobile Robotics Laboratory

Project theme:

**Electronic Instrumentation of an Educational Bench for Angular Aero Control of Multirotors**

Keywords: Drones, Instrumentation, Sensors, Control Bench, Computer Vision.

## Research Team

- Candidate: Eduarda Rodrigues Sproesser
- Supervising Professor: André Carmona Hernandes
- Co-supervising Professor: Marcelo Becker
- Scholarship Supervisor Abroad: Stefano Mintchev

## Abstract (General)

This work reports the development and evaluation of optical tactile sensing strategies for collision-resilient aerial platforms, conducted during a BEPE internship at the Environmental Robotics Laboratory (ERL), ETH Zurich.

The study investigates whether non-planar marker configurations can reduce geometric ambiguities typically associated with planar optical sensors. Five marker topologies were designed and experimentally evaluated (one planar baseline and four non-planar configurations), using an automated setup that controls position and orientation and enables direct comparison between estimated and ground-truth poses.

Results show trade-offs rather than uniform superiority of non-planar geometries. The planar baseline (`1p`) achieved the best mean translation accuracy (3.80 mm), while non-planar configurations improved robustness in specific conditions. The edge-connected configuration (`2e`) achieved the highest detection rate (0.77), and for vertex-connected markers (`3v`) the multi-iterative solver reduced rotation error (from 29.02° to 17.00°), with increased translation noise due to marker miniaturization and lower pixel resolution.

Overall, non-planar geometries are promising for applications requiring angular robustness and detection continuity under deformation, while planar markers may remain advantageous for translation precision in constrained imaging conditions.

## Repository Guide

This is a top-level overview. Each subfolder contains its own detailed README with usage instructions.

- `calib_cam/`: camera calibration utilities (image capture, calibration scripts, marker PDF generation).
- `collect_data/`: automated experiment/data collection pipeline (printer motion, LED control, capture orchestration).
- `pose/`: marker corner extraction, pose estimation, plotting, and error analysis.

## Media and Supporting Material

CAD files, videos, presentations, and photos are stored in the **`mídia`** folder.

## Notes

- The project includes scripts for real hardware workflows; review folder-level READMEs before execution.
- Path configuration is environment-variable based in key scripts for portability across machines.
