---
title: "PET Image Reconstruction and Deformable Motion Correction Using Unorganized Point Clouds"
collection: publications
permalink: /publication/2015-10-01-paper-title-number-7
excerpt: 'We propose an alternative approach to iterative image reconstruction with correction for deformable motion, wherein unorganized point clouds are used to model the imaged objects in the image space, and motion is corrected for explicitly by introducing a time-dependence into the point coordinates.'
date: 2017-03-02
venue: 'IEEE Transactions on Medical Imaging'
paperurl: 'https://ieeexplore.ieee.org/document/7866817'
citation: 'I. S. Klyuzhin, V. Sossi (2017). &quot;PET Image Reconstruction and Deformable Motion Correction Using Unorganized Point Clouds.&quot; <i>IEEE Trans. Med. Imag.</i>, 36(6).'
---
Quantitative positron emission tomography imaging often requires correcting the image data for deformable motion. With cyclic motion, this is traditionally achieved by separating the coincidence data into a relatively small number of gates, and incorporating the inter-gate image transformation matrices into the reconstruction algorithm. In the presence of non-cyclic deformable motion, this approach may be impractical due to a large number of required gates. In this paper, we propose an alternative approach to iterative image reconstruction with correction for deformable motion, wherein unorganized point clouds are used to model the imaged objects in the image space, and motion is corrected for explicitly by introducing a time-dependence into the point coordinates. The image function is represented using constant basis functions with finite support determined by the boundaries of the Voronoi cells in the point cloud. We validate the quantitative accuracy and stability of the proposed approach by reconstructing noise-free and noisy projection data from digital and physical phantoms. The point-cloud-based maximum likelihood expectation maximization (MLEM) and one-pass list-mode ordered-subset expectation maximization (OSEM) algorithms are validated. The results demonstrate that images reconstructed using the proposed method are quantitatively stable, with noise and convergence properties comparable to image reconstruction based on the use of rectangular and radially-symmetric basis functions.
<!-- 
[Download paper here](http://academicpages.github.io/files/paper3.pdf)

Recommended citation: Your Name, You. (2015). "Paper Title Number 3." <i>Journal 1</i>. 1(3).
 -->