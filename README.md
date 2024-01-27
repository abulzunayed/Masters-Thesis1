# Name of Thesis: Improved Magnetic field mapping based on DREAM MRI and physics informed Neural Network.

Magnetic Resonance Imaging (MRI) relies on the main static magnetic ϐield (B0) and the
radiofrequency ϐield (B1) for spin polarization and magnetization. The ultrafast Dual
Refocusing Echo Acquisition Mode (DREAM) sequence is proceeded to generate B0 and
B1 maps where it’s able to cover the whole transmit coil volume in only one second, which
is more than an order of magnitude faster than established approaches. This established
technique provides quite promising B0 and B1 maps. However, small significant
deviations had been addressed in B0 and B1 maps which need to be corrected. Changing
the internal parameters of the sequences is not worthy and simple where the DREAM
sequence provides short acquisition time (a few seconds per slice). Therefore, it is needed
to post-process of magnetic ϐield maps B0 and B1. Integration of the machine learning
approaches can resolve these implicit challenges. Speciϐically, proposing by two novel
methods: Multiple linear projections model and Artiϐicial neural network model. Both
models minimize the fraction of the B0 and B1 maps simultaneously it is possible to
reduce the divergence in the quality of both maps. After the implementation of both
models in my thesis work, the comparative analysis reveals that the Multiple Linear
Projections model yields considerably accurate B0 maps, while Artiϐicial Neural Network
model emerges as a more promising B1 map due to its marginal computation of data
weights. Eventually, the optimal combination of both models effectively addresses the
divergence in the quality of both B0 and B1 maps and serves as the additional unique
solution for the DREAM sequence.

1. Introduction
In MRI, the main static magnetic ϐield referred to as B0 and B1 ϐield is produced either by
a local coil or more commonly, from windings in the wall of the scanner itself. B0 is the
constant homogeneous magnetic ϐield used to polarize spins that create magnetization. It
can refer to both the direction and the strength of the ϐield. The RF ϐield (B1) is applied
perpendicular to the longitudinal axis (B0) to agitate the magnetization in some manner
(e.g. excitation pulses, inversion pulses, etc). The inhomogeneity of magnetic fields
(mainly B0 and B1) is a common problem in MRI. Although the scanners and coils are
specially specified to produced maximum homogeneity, some inhomogeneity in the fields
remains. As B0 and B1 are affected by the magnetic susceptibility of the subject, the
inhomogeneity in magnetic fields (B0 and B1) leads to artifacts to the optimal image
quality. In response, many different methods have been developed to acquire B0 and B1
maps and dual refocusing echo acquisition mode (DREAM) sequence is one of them.
DREAM sequence works as an ultrafast MRI technique which simultaneously acquires
two images, enabling the calculation of B1, B0. After the acquisition of the sequence, it needs
to be corrected and enhanced B0 and B1 maps to overcome the inhomogeneity effects.
DREAM sequence introduced by Nehrke et al. [ 3 ] in 2012 which relies on a stimulated
echo acquisition mode (STEAM) preparation sequence followed by a customized single-shot gradient echo sequence. This approach simultaneously measures the stimulated
echo and the free induction decay as gradient-recalled echoes. Afterward, the actual ϐlip
angle of the STEAM preparation radiofrequency pulses is determined from the ratio of the
two measured signals. Moreover, again Tim Baum explored the refinement of the DREAM
sequence’s B0 and B1 maps which was implemented in his work [7]. Two strategies were followed to generate B0 and B1 maps: STE-first and STE-ID-first that are quite
sound to optimize the sequence parameters and image quality. However, small significant
deviations had been addressed in B0 and B1 maps which need to be corrected. In order
to do that, changing the internal parameters of the sequences is not worthy and simple
where the DREAM sequence provides a short acquisition time (a few seconds per slice)
without relying on low-bandwidth acquisition (e.g EPI or spirals) [ 18] that are prone to
geometric distortion and /or blurring caused by inhomogeneities of the static magnetic field.
Therefore, it is needed for the post-processing of magnetic field maps B0 and B1.Abul Kasem Mohammad Zunayed
8
The journey of AI started a long time ago in 1956 at a workshop at Dartmouth College
[17]. Recently machine learning, a subset of AI, has become widely recognized to solve
the MRI-related tasks such as image acquisition, enhancement, reconstruction, denoising,
prediction and so on. In the DREAM sequence, pursuing the optimality of B0 and B1 maps
which was implemented by Tim Baum [7], addressed slight variations in the image
quality. In my thesis works, I try to minimize these deviations of B0 and B1 maps by
applying machine learning approaches. In response, I proposed two methods to improve
the quality of the B0 and B1 maps: a multiple linear projections model and an artiϐicial
neural network model. In addition, these two models have been validated by the WASABI
sequence compared with the produced invivo B0 and B1 maps from the DREAM sequence.
![image](https://github.com/abulzunayed/Masters-Thesis/assets/122612945/281cdc43-9bd9-4bcc-96b7-a064e0e9a03e)
![image](https://github.com/abulzunayed/Masters-Thesis/assets/122612945/6d342af2-1e0b-4f2a-b207-1d5fda90955c)

6. Conclusion
The suggested two methods Multiple linear projections model and the Artiϐicial neural
network model serve as the additional unique solution for the DREAM sequence to
diminish the fraction of B0 and B1 maps and simultaneously minimize the divergence in the
quality of both maps. Exclusively, this is achieved without any disruption from internal
sequence parameters. Even though the Multiple linear projections model yields a
considerably accurate B1 map, the Artiϐicial neural Network model seems more
promising. Because in the artiϐicial neural network model, the weight of each data was
computed marginal way. Moreover, through comparison between both models, I can
conclude that the Multiple linear projections model is the best ϐitted for B0 maps whereas
the Artiϐicial neural network model implies of superior performance in generating B1
maps. This observation is substantiated by the evidence in Table5(refer to subsection
4.3.1). Therefore, the best technique may be the combination of B0 with linear projection
and B1 with NN model. In the future, the generalization of the proposed methods should
be investigated because many hyper-parameters are involved within these two models.

