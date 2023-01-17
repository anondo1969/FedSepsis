# FedSepsis

**[FedSepsis: A Federated Multi-Modal Deep Learning-Based Internet of Medical Things Application for Early Detection of Sepsis from Electronic Health Records Using Raspberry Pi and Jetson Nano Devices](https://doi.org/10.3390/s23020970)**, Mahbub Ul Alam, Rahim Rahmani. Sensors 23, no. 2: 970, https://doi.org/10.3390/s23020970. (**[presentation slides](http://mahbub.blogs.dsv.su.se/files/2023/01/FedSepsis_Mahbub_Ul_Alam.pdf)**)

## Abstract

 The concept of the Internet of Medical Things brings a promising option to utilize various electronic health records stored in different medical devices and servers to create practical but secure clinical decision support systems. To achieve such a system, we need to focus on several aspects, most notably the usability aspect of deploying it using low-end devices. This study introduces one such application, namely FedSepsis, for the early detection of sepsis using electronic health records. We incorporate several cutting-edge deep learning techniques for the prediction and natural-language processing tasks. We also explore the multimodality aspect for the better use of electronic health records. A secure distributed machine learning mechanism is essential to building such a practical internet of medical things application. To address this, we analyze two federated learning techniques. Moreover, we use two different kinds of low-computational edge devices, namely Raspberry Pi and Jetson Nano, to address the challenges of using such a system in a practical setting and report the comparisons. We report several critical system-level information about the devices, namely CPU utilization, disk utilization, process CPU threads in use, process memory in use (non-swap), process memory available (non-swap), system memory utilization, temperature, and network traffic. We publish the prediction results with the evaluation metrics area under the receiver operating characteristic curve, the area under the precision–recall curve, and the earliness to predict sepsis in hours. Our results show that the performance is satisfactory, and with a moderate amount of devices, the federated learning setting results are similar to the single server-centric setting. Multimodality provides the best results compared to any single modality in the input features obtained from the electronic health records. Generative adversarial neural networks provide a clear superiority in handling the sparsity of electronic health records. Multimodality with the generative adversarial neural networks provides the best result: the area under the precision–recall curve is 96.55%, the area under the receiver operating characteristic curve is 99.35%, and earliness is 4.56 h. FedSepsis suggests that incorporating such a concept together with low-end computational devices could be beneficial for all the medical sector stakeholders and should be explored further.

### Schematic diagram of FedSepsis

![Schematic diagram of FedSepsis](http://mahbub.blogs.dsv.su.se/files/2023/01/neural_network_diagram.png)

## Citation

Please acknowledge the following work in papers or derivative software:

Alam MU, Rahmani R. FedSepsis: A Federated Multi-Modal Deep Learning-Based Internet of Medical Things Application for Early Detection of Sepsis from Electronic Health Records Using Raspberry Pi and Jetson Nano Devices. Sensors. 2023; 23(2):970. https://doi.org/10.3390/s23020970

### Bibtex Format for Citation

```
  @article{alam2023,
  author={Alam, Mahbub Ul and Rahmani, Rahim},
  title={FedSepsis: A Federated Multi-Modal Deep Learning-Based Internet of Medical Things Application for Early Detection of Sepsis from Electronic Health Records Using Raspberry Pi and Jetson Nano Devices}, 
  volume={23}, ISSN={1424-8220}, 
  url={http://dx.doi.org/10.3390/s23020970}, 
  DOI={10.3390/s23020970}, number={2}, journal={Sensors}, 
  publisher={MDPI AG},  
  year={2023}, 
  month={Jan}, 
  pages={970} 
  }
```


