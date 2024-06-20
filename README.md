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

![image](https://github.com/abulzunayed/Masters_Thesis-_Create-NN-model-for-MRI-sequence-Image-and-Postprocessing/assets/122612945/525e5ab7-04b7-4940-a298-eb97879bf653)

![image](https://github.com/abulzunayed/Masters_Thesis-NN-model-for-MRI-sequence-Image-and-Postprocessing/assets/122612945/74b46854-2b93-4ec5-a686-cce18bdae128)


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



