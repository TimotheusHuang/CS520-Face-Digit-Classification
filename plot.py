# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt

perceptron_digits_list = [[11.435525894165039, 0.715, 0.28500000000000003], [25.93780493736267, 0.753, 0.247], [39.46669101715088, 0.768, 0.23199999999999998], [52.22143888473511, 0.75, 0.25], [64.88742899894714, 0.735, 0.265], [84.6507179737091, 0.782, 0.21799999999999997], [96.95536518096924, 0.76, 0.24], [105.75979614257812, 0.78, 0.21999999999999997], [119.97196292877197, 0.762, 0.238], [143.11764192581177, 0.784, 0.21599999999999997], [14.33385682106018, 0.685, 0.31499999999999995], [26.883075952529907, 0.778, 0.22199999999999998], [41.6833291053772, 0.741, 0.259], [54.59540414810181, 0.721, 0.279], [67.76258587837219, 0.756, 0.244], [86.00167608261108, 0.751, 0.249], [94.99104690551758, 0.737, 0.263], [110.12046313285828, 0.802, 0.19799999999999995], [122.58913588523865, 0.769, 0.23099999999999998], [137.81235194206238, 0.799, 0.20099999999999996], [19.732338905334473, 0.746, 0.254], [41.14393901824951, 0.742, 0.258], [58.23711395263672, 0.714, 0.28600000000000003], [79.09326410293579, 0.764, 0.236], [117.81723189353943, 0.802, 0.19799999999999995], [131.09787321090698, 0.75, 0.25], [147.97750091552734, 0.75, 0.25], [171.9693419933319, 0.791, 0.20899999999999996], [176.412202835083, 0.797, 0.20299999999999996], [138.19983100891113, 0.826, 0.17400000000000004], [16.62510085105896, 0.696, 0.30400000000000005], [26.79927897453308, 0.774, 0.22599999999999998], [40.28752613067627, 0.705, 0.29500000000000004], [52.032768964767456, 0.795, 0.20499999999999996], [66.57323408126831, 0.749, 0.251], [79.5028829574585, 0.747, 0.253], [95.40633702278137, 0.793, 0.20699999999999996], [105.65664601325989, 0.764, 0.236], [128.35896801948547, 0.808, 0.19199999999999995], [133.0942280292511, 0.757, 0.243], [13.636476993560791, 0.746, 0.254], [26.32023787498474, 0.796, 0.20399999999999996], [39.84436297416687, 0.761, 0.239], [52.85856080055237, 0.795, 0.20499999999999996], [66.50201082229614, 0.748, 0.252], [84.11007905006409, 0.773, 0.22699999999999998], [92.27373194694519, 0.79, 0.20999999999999996], [105.61292004585266, 0.776, 0.22399999999999998], [119.79902482032776, 0.82, 0.18000000000000005], [133.32151103019714, 0.813, 0.18700000000000006]]
perceptron_digits_list_5 = [[23.5830659866333, 0.701, 0.29900000000000004], [45.11157298088074, 0.802, 0.19799999999999995], [67.01315498352051, 0.759, 0.241], [94.41755485534668, 0.744, 0.256], [113.02034783363342, 0.776, 0.22399999999999998], [171.3111231327057, 0.766, 0.23399999999999999], [160.76866793632507, 0.793, 0.20699999999999996], [179.0214078426361, 0.791, 0.20899999999999996], [202.07851481437683, 0.793, 0.20699999999999996], [224.82105588912964, 0.804, 0.19599999999999995], [21.971791982650757, 0.731, 0.269], [45.49067401885986, 0.765, 0.235], [70.7936339378357, 0.794, 0.20599999999999996], [94.53760600090027, 0.747, 0.253], [114.69485688209534, 0.749, 0.251], [186.7987780570984, 0.777, 0.22299999999999998], [301.4623019695282, 0.792, 0.20799999999999996], [347.10013818740845, 0.795, 0.20499999999999996], [299.96253085136414, 0.805, 0.19499999999999995], [332.9611859321594, 0.76, 0.24], [21.90148401260376, 0.698, 0.30200000000000005], [47.7346727848053, 0.776, 0.22399999999999998], [65.51240801811218, 0.78, 0.21999999999999997], [89.8886239528656, 0.768, 0.23199999999999998], [111.84716391563416, 0.722, 0.278], [136.5541889667511, 0.792, 0.20799999999999996], [155.45950293540955, 0.78, 0.21999999999999997], [173.06893396377563, 0.726, 0.274], [199.76450204849243, 0.752, 0.248], [228.0732982158661, 0.796, 0.20399999999999996], [25.473644018173218, 0.712, 0.28800000000000003], [44.46287393569946, 0.774, 0.22599999999999998], [64.06927609443665, 0.765, 0.235], [86.6128580570221, 0.791, 0.20899999999999996], [109.07936501502991, 0.779, 0.22099999999999997], [135.6159269809723, 0.777, 0.22299999999999998], [156.147136926651, 0.726, 0.274], [174.83551692962646, 0.734, 0.266], [211.01104497909546, 0.782, 0.21799999999999997], [220.87261509895325, 0.793, 0.20699999999999996], [21.844370126724243, 0.762, 0.238], [47.33081102371216, 0.786, 0.21399999999999997], [66.45968389511108, 0.733, 0.267], [92.96804785728455, 0.795, 0.20499999999999996], [108.61449003219604, 0.782, 0.21799999999999997], [131.75993704795837, 0.782, 0.21799999999999997], [156.30486392974854, 0.81, 0.18999999999999995], [184.09595394134521, 0.766, 0.23399999999999999], [199.5915608406067, 0.819, 0.18100000000000005], [221.08165502548218, 0.81, 0.18999999999999995]]
nb_digits_list = [[0.13544201850891113, 0.745, 0.255], [0.2727088928222656, 0.782, 0.21799999999999997], [0.37746095657348633, 0.788, 0.21199999999999997], [0.5236690044403076, 0.759, 0.241], [0.7080099582672119, 0.782, 0.21799999999999997], [0.7144758701324463, 0.776, 0.22399999999999998], [0.8309791088104248, 0.765, 0.235], [0.9592499732971191, 0.771, 0.22899999999999998], [1.552170991897583, 0.767, 0.23299999999999998], [1.220818042755127, 0.774, 0.22599999999999998], [0.19325709342956543, 0.757, 0.243], [0.3722109794616699, 0.769, 0.23099999999999998], [0.5593349933624268, 0.774, 0.22599999999999998], [0.8791089057922363, 0.771, 0.22899999999999998], [0.9448020458221436, 0.78, 0.21999999999999997], [1.3350880146026611, 0.769, 0.23099999999999998], [1.522529125213623, 0.773, 0.22699999999999998], [1.5570437908172607, 0.765, 0.235], [1.8524389266967773, 0.767, 0.23299999999999998], [1.8974149227142334, 0.774, 0.22599999999999998], [0.17435503005981445, 0.764, 0.236], [0.23849105834960938, 0.757, 0.243], [0.36349010467529297, 0.757, 0.243], [0.47557687759399414, 0.759, 0.241], [0.5899920463562012, 0.769, 0.23099999999999998], [0.8074629306793213, 0.77, 0.22999999999999998], [0.8287150859832764, 0.766, 0.23399999999999999], [0.9607510566711426, 0.768, 0.23199999999999998], [1.0729238986968994, 0.768, 0.23199999999999998], [1.2016549110412598, 0.774, 0.22599999999999998], [0.11921405792236328, 0.757, 0.243], [0.2439570426940918, 0.771, 0.22899999999999998], [0.36226487159729004, 0.773, 0.22699999999999998], [0.4783749580383301, 0.764, 0.236], [0.5948638916015625, 0.775, 0.22499999999999998], [0.7127330303192139, 0.766, 0.23399999999999999], [0.8278858661651611, 0.773, 0.22699999999999998], [0.9325039386749268, 0.773, 0.22699999999999998], [1.1194090843200684, 0.773, 0.22699999999999998], [1.1801888942718506, 0.774, 0.22599999999999998], [0.12191390991210938, 0.775, 0.22499999999999998], [0.24162507057189941, 0.771, 0.22899999999999998], [0.3658308982849121, 0.765, 0.235], [0.477247953414917, 0.769, 0.23099999999999998], [0.5907948017120361, 0.777, 0.22299999999999998], [0.7378108501434326, 0.771, 0.22899999999999998], [0.8394510746002197, 0.774, 0.22599999999999998], [0.9500339031219482, 0.773, 0.22699999999999998], [1.0860249996185303, 0.77, 0.22999999999999998], [1.1945691108703613, 0.774, 0.22599999999999998]]
kNN_digits_list = [[1.5610110759735107, 0.799, 0.20099999999999996], [2.844533920288086, 0.834, 0.16600000000000004], [4.202785015106201, 0.864, 0.136], [5.375244140625, 0.869, 0.131], [6.774034023284912, 0.88, 0.12], [8.06804084777832, 0.879, 0.121], [9.506840944290161, 0.884, 0.11599999999999999], [10.726548910140991, 0.9, 0.09999999999999998], [12.055723905563354, 0.901, 0.09899999999999998], [13.434770107269287, 0.903, 0.09699999999999998], [3.00101900100708, 0.758, 0.242], [5.388073921203613, 0.825, 0.17500000000000004], [7.4493489265441895, 0.853, 0.14700000000000002], [9.961136102676392, 0.871, 0.129], [12.435633897781372, 0.885, 0.11499999999999999], [14.43146300315857, 0.885, 0.11499999999999999], [16.888336181640625, 0.897, 0.10299999999999998], [18.599955081939697, 0.896, 0.10399999999999998], [20.541703939437866, 0.899, 0.10099999999999998], [23.39443612098694, 0.907, 0.09299999999999997], [1.5664119720458984, 0.788, 0.21199999999999997], [2.767136812210083, 0.843, 0.15700000000000003], [3.940739154815674, 0.858, 0.14200000000000002], [5.240215063095093, 0.862, 0.138], [6.570421934127808, 0.885, 0.11499999999999999], [7.73821496963501, 0.896, 0.10399999999999998], [9.024922847747803, 0.895, 0.10499999999999998], [10.254048109054565, 0.888, 0.11199999999999999], [11.652408123016357, 0.898, 0.10199999999999998], [13.022999048233032, 0.908, 0.09199999999999997], [1.5530040264129639, 0.789, 0.21099999999999997], [2.8220720291137695, 0.842, 0.15800000000000003], [3.920483112335205, 0.841, 0.15900000000000003], [5.208721160888672, 0.871, 0.129], [6.469344854354858, 0.888, 0.11199999999999999], [7.851212024688721, 0.879, 0.121], [9.079951047897339, 0.887, 0.11299999999999999], [10.397702932357788, 0.894, 0.10599999999999998], [11.690814018249512, 0.903, 0.09699999999999998], [13.007493019104004, 0.904, 0.09599999999999997], [1.5537898540496826, 0.779, 0.22099999999999997], [2.765089988708496, 0.835, 0.16500000000000004], [3.9388699531555176, 0.864, 0.136], [5.408725023269653, 0.882, 0.118], [6.527618885040283, 0.886, 0.11399999999999999], [7.828556060791016, 0.88, 0.12], [9.014636993408203, 0.895, 0.10499999999999998], [10.385730981826782, 0.893, 0.10699999999999998], [11.669586896896362, 0.904, 0.09599999999999997], [12.927479982376099, 0.905, 0.09499999999999997]]
kNN_digits_list_10 = [[1.5667669773101807, 0.767, 0.23299999999999998], [2.993129014968872, 0.815, 0.18500000000000005], [4.137798070907593, 0.845, 0.15500000000000003], [5.404545068740845, 0.859, 0.14100000000000001], [6.8708717823028564, 0.862, 0.138], [7.986776828765869, 0.874, 0.126], [9.353160858154297, 0.887, 0.11299999999999999], [11.130709171295166, 0.886, 0.11399999999999999], [12.154846906661987, 0.894, 0.10599999999999998], [13.299469947814941, 0.894, 0.10599999999999998], [3.187187910079956, 0.724, 0.276], [5.307278156280518, 0.809, 0.19099999999999995], [7.544886112213135, 0.83, 0.17000000000000004], [9.585383176803589, 0.851, 0.14900000000000002], [12.271965026855469, 0.871, 0.129], [14.524586915969849, 0.876, 0.124], [16.420989990234375, 0.886, 0.11399999999999999], [19.982055187225342, 0.884, 0.11599999999999999], [21.949748992919922, 0.882, 0.118], [27.369163990020752, 0.891, 0.10899999999999999], [1.5080301761627197, 0.768, 0.23199999999999998], [2.838408946990967, 0.821, 0.17900000000000005], [3.974947929382324, 0.831, 0.16900000000000004], [5.219620943069458, 0.863, 0.137], [6.629104852676392, 0.874, 0.126], [7.830566167831421, 0.877, 0.123], [9.043948888778687, 0.883, 0.11699999999999999], [10.325039863586426, 0.886, 0.11399999999999999], [11.648117065429688, 0.89, 0.10999999999999999], [12.996098041534424, 0.893, 0.10699999999999998], [1.516503095626831, 0.706, 0.29400000000000004], [2.8354110717773438, 0.831, 0.16900000000000004], [3.961606979370117, 0.846, 0.15400000000000003], [5.23025107383728, 0.869, 0.131], [6.567314863204956, 0.861, 0.139], [7.825016975402832, 0.881, 0.119], [9.064451932907104, 0.877, 0.123], [10.366902828216553, 0.891, 0.10899999999999999], [11.641438007354736, 0.892, 0.10799999999999998], [13.085409164428711, 0.892, 0.10799999999999998], [1.503161907196045, 0.748, 0.252], [2.8420920372009277, 0.811, 0.18899999999999995], [3.988607168197632, 0.838, 0.16200000000000003], [5.269127130508423, 0.853, 0.14700000000000002], [6.58343505859375, 0.86, 0.14], [7.795169115066528, 0.875, 0.125], [9.096771001815796, 0.892, 0.10799999999999998], [10.530802011489868, 0.889, 0.11099999999999999], [11.680070877075195, 0.889, 0.11099999999999999], [12.96094012260437, 0.895, 0.10499999999999998]]
perceptron_faces_list = [[1.6930491924285889, 0.7666666666666667, 0.23333333333333328], [3.274935007095337, 0.7733333333333333, 0.22666666666666668], [4.62597918510437, 0.7133333333333334, 0.2866666666666666], [6.104032039642334, 0.8533333333333334, 0.1466666666666666], [7.7103190422058105, 0.8066666666666666, 0.19333333333333336], [8.953853845596313, 0.8666666666666667, 0.1333333333333333], [10.373960971832275, 0.86, 0.14], [12.931293964385986, 0.84, 0.16000000000000003], [13.187888860702515, 0.8266666666666667, 0.17333333333333334], [14.674576044082642, 0.8733333333333333, 0.1266666666666667], [3.4349119663238525, 0.7933333333333333, 0.20666666666666667], [6.603621006011963, 0.86, 0.14], [9.232633829116821, 0.8, 0.19999999999999996], [9.452639102935791, 0.8266666666666667, 0.17333333333333334], [12.263037919998169, 0.7666666666666667, 0.23333333333333328], [13.880486011505127, 0.7866666666666666, 0.21333333333333337], [15.331003904342651, 0.8666666666666667, 0.1333333333333333], [17.812133073806763, 0.88, 0.12], [19.852911949157715, 0.8933333333333333, 0.10666666666666669], [22.380132913589478, 0.8333333333333334, 0.16666666666666663], [1.6995069980621338, 0.7, 0.30000000000000004], [3.205230951309204, 0.7933333333333333, 0.20666666666666667], [4.57515811920166, 0.8133333333333334, 0.18666666666666665], [6.028670072555542, 0.7733333333333333, 0.22666666666666668], [7.263846158981323, 0.7933333333333333, 0.20666666666666667], [8.89138412475586, 0.8, 0.19999999999999996], [10.041184902191162, 0.9, 0.09999999999999998], [12.439428091049194, 0.8666666666666667, 0.1333333333333333], [12.75128984451294, 0.8666666666666667, 0.1333333333333333], [14.116371870040894, 0.8666666666666667, 0.1333333333333333], [1.632800817489624, 0.7266666666666667, 0.2733333333333333], [3.099025011062622, 0.84, 0.16000000000000003], [4.570297956466675, 0.7466666666666667, 0.2533333333333333], [5.984944820404053, 0.8533333333333334, 0.1466666666666666], [7.120076894760132, 0.8066666666666666, 0.19333333333333336], [8.818700075149536, 0.8666666666666667, 0.1333333333333333], [11.184056997299194, 0.8466666666666667, 0.15333333333333332], [11.522845029830933, 0.8333333333333334, 0.16666666666666663], [12.72283411026001, 0.86, 0.14], [14.018379926681519, 0.8733333333333333, 0.1266666666666667], [1.6346080303192139, 0.78, 0.21999999999999997], [3.227092981338501, 0.7666666666666667, 0.23333333333333328], [4.559140920639038, 0.7866666666666666, 0.21333333333333337], [5.956784963607788, 0.8666666666666667, 0.1333333333333333], [7.356001138687134, 0.84, 0.16000000000000003], [8.752471923828125, 0.84, 0.16000000000000003], [10.043222904205322, 0.8866666666666667, 0.11333333333333329], [11.495483875274658, 0.8533333333333334, 0.1466666666666666], [12.821668863296509, 0.8866666666666667, 0.11333333333333329], [14.225602149963379, 0.8666666666666667, 0.1333333333333333]]
perceptron_faces_list_5 = [[2.3791208267211914, 0.7466666666666667, 0.2533333333333333], [4.868394136428833, 0.7666666666666667, 0.23333333333333328], [7.512495040893555, 0.8133333333333334, 0.18666666666666665], [9.978740930557251, 0.8, 0.19999999999999996], [11.656368017196655, 0.8266666666666667, 0.17333333333333334], [14.126967906951904, 0.8466666666666667, 0.15333333333333332], [16.437583923339844, 0.8333333333333334, 0.16666666666666663], [21.810767889022827, 0.86, 0.14], [20.945842027664185, 0.88, 0.12], [23.33459210395813, 0.88, 0.12], [3.736220121383667, 0.78, 0.21999999999999997], [7.217383861541748, 0.86, 0.14], [11.293092012405396, 0.8533333333333334, 0.1466666666666666], [15.298814058303833, 0.82, 0.18000000000000005], [19.326346158981323, 0.8333333333333334, 0.16666666666666663], [23.22660207748413, 0.8533333333333334, 0.1466666666666666], [25.81170392036438, 0.8866666666666667, 0.11333333333333329], [29.352849006652832, 0.8533333333333334, 0.1466666666666666], [32.75386309623718, 0.84, 0.16000000000000003], [39.695966958999634, 0.8933333333333333, 0.10666666666666669], [2.3709850311279297, 0.7533333333333333, 0.2466666666666667], [4.757345914840698, 0.78, 0.21999999999999997], [6.901926040649414, 0.82, 0.18000000000000005], [9.148207902908325, 0.8266666666666667, 0.17333333333333334], [11.283878087997437, 0.8533333333333334, 0.1466666666666666], [14.679362058639526, 0.8533333333333334, 0.1466666666666666], [15.743775129318237, 0.88, 0.12], [17.991941928863525, 0.8666666666666667, 0.1333333333333333], [20.12490177154541, 0.8666666666666667, 0.1333333333333333], [22.384777069091797, 0.8533333333333334, 0.1466666666666666], [2.3924169540405273, 0.6466666666666666, 0.3533333333333334], [4.668341875076294, 0.82, 0.18000000000000005], [7.29784893989563, 0.8133333333333334, 0.18666666666666665], [9.09635305404663, 0.8266666666666667, 0.17333333333333334], [11.399417877197266, 0.8066666666666666, 0.19333333333333336], [13.520313024520874, 0.8666666666666667, 0.1333333333333333], [18.333466053009033, 0.84, 0.16000000000000003], [18.041232109069824, 0.8733333333333333, 0.1266666666666667], [22.12174892425537, 0.86, 0.14], [22.338162899017334, 0.9066666666666666, 0.09333333333333338], [2.3419010639190674, 0.62, 0.38], [4.716040134429932, 0.8133333333333334, 0.18666666666666665], [6.885394096374512, 0.8133333333333334, 0.18666666666666665], [10.034198999404907, 0.8466666666666667, 0.15333333333333332], [11.456454992294312, 0.8266666666666667, 0.17333333333333334], [13.51309084892273, 0.8466666666666667, 0.15333333333333332], [15.574290990829468, 0.86, 0.14], [17.839334964752197, 0.7866666666666666, 0.21333333333333337], [20.012650966644287, 0.8866666666666667, 0.11333333333333329], [22.237930059432983, 0.8933333333333333, 0.10666666666666669]]
nb_faces_list = [[0.27846312522888184, 0.72, 0.28], [0.5398991107940674, 0.8733333333333333, 0.1266666666666667], [0.809938907623291, 0.86, 0.14], [1.0743541717529297, 0.9066666666666666, 0.09333333333333338], [1.3451299667358398, 0.8666666666666667, 0.1333333333333333], [1.6110539436340332, 0.8733333333333333, 0.1266666666666667], [1.8877902030944824, 0.8733333333333333, 0.1266666666666667], [2.160485029220581, 0.9, 0.09999999999999998], [2.4352619647979736, 0.8933333333333333, 0.10666666666666669], [2.698012113571167, 0.8866666666666667, 0.11333333333333329], [0.4906799793243408, 0.64, 0.36], [0.8667511940002441, 0.8066666666666666, 0.19333333333333336], [1.2163410186767578, 0.8133333333333334, 0.18666666666666665], [1.5727019309997559, 0.8333333333333334, 0.16666666666666663], [2.018167018890381, 0.8666666666666667, 0.1333333333333333], [2.4297561645507812, 0.8466666666666667, 0.15333333333333332], [2.779208183288574, 0.8866666666666667, 0.11333333333333329], [3.189379930496216, 0.9, 0.09999999999999998], [3.5556371212005615, 0.8666666666666667, 0.1333333333333333], [3.8253040313720703, 0.8866666666666667, 0.11333333333333329], [0.2711009979248047, 0.8, 0.19999999999999996], [0.5345439910888672, 0.82, 0.18000000000000005], [0.7926340103149414, 0.8733333333333333, 0.1266666666666667], [1.0525808334350586, 0.8733333333333333, 0.1266666666666667], [1.3251349925994873, 0.84, 0.16000000000000003], [1.575429916381836, 0.8733333333333333, 0.1266666666666667], [1.8454201221466064, 0.8933333333333333, 0.10666666666666669], [2.1013400554656982, 0.8733333333333333, 0.1266666666666667], [2.336716890335083, 0.88, 0.12], [2.8257060050964355, 0.8866666666666667, 0.11333333333333329], [0.27437400817871094, 0.7933333333333333, 0.20666666666666667], [0.530695915222168, 0.7666666666666667, 0.23333333333333328], [0.8041069507598877, 0.86, 0.14], [1.056825876235962, 0.8666666666666667, 0.1333333333333333], [1.3260538578033447, 0.88, 0.12], [1.58827805519104, 0.88, 0.12], [1.8446509838104248, 0.8733333333333333, 0.1266666666666667], [2.1248669624328613, 0.8733333333333333, 0.1266666666666667], [2.3493380546569824, 0.8866666666666667, 0.11333333333333329], [2.863805055618286, 0.8866666666666667, 0.11333333333333329], [0.27082395553588867, 0.5133333333333333, 0.4866666666666667], [0.5356659889221191, 0.8466666666666667, 0.15333333333333332], [0.8029749393463135, 0.86, 0.14], [1.0578758716583252, 0.8666666666666667, 0.1333333333333333], [1.331502914428711, 0.88, 0.12], [1.5869669914245605, 0.86, 0.14], [1.8719310760498047, 0.88, 0.12], [2.118095874786377, 0.9, 0.09999999999999998], [2.3726720809936523, 0.8866666666666667, 0.11333333333333329], [2.876215934753418, 0.8866666666666667, 0.11333333333333329]]
kNN_faces_list = [[1.158195972442627, 0.5533333333333333, 0.44666666666666666], [1.0621240139007568, 0.5133333333333333, 0.4866666666666667], [1.389159917831421, 0.6733333333333333, 0.32666666666666666], [1.258965015411377, 0.6933333333333334, 0.30666666666666664], [1.3769810199737549, 0.7, 0.30000000000000004], [1.4939930438995361, 0.5133333333333333, 0.4866666666666667], [1.568208932876587, 0.7, 0.30000000000000004], [1.6734898090362549, 0.7333333333333333, 0.2666666666666667], [1.7544629573822021, 0.5733333333333334, 0.42666666666666664], [1.8866569995880127, 0.5666666666666667, 0.43333333333333335], [1.4280622005462646, 0.5333333333333333, 0.4666666666666667], [1.5596799850463867, 0.5866666666666667, 0.41333333333333333], [1.7611591815948486, 0.6133333333333333, 0.3866666666666667], [1.8294320106506348, 0.5133333333333333, 0.4866666666666667], [2.3413960933685303, 0.6066666666666667, 0.3933333333333333], [2.176597833633423, 0.66, 0.33999999999999997], [2.6496260166168213, 0.5533333333333333, 0.44666666666666666], [2.752774953842163, 0.58, 0.42000000000000004], [2.9493300914764404, 0.56, 0.43999999999999995], [2.9104678630828857, 0.5733333333333334, 0.42666666666666664], [1.141535997390747, 0.62, 0.38], [1.0495588779449463, 0.5133333333333333, 0.4866666666666667], [1.1558640003204346, 0.6, 0.4], [1.4465689659118652, 0.5666666666666667, 0.43333333333333335], [1.3327610492706299, 0.5466666666666666, 0.45333333333333337], [1.4663949012756348, 0.5533333333333333, 0.44666666666666666], [1.750074863433838, 0.5533333333333333, 0.44666666666666666], [1.6678900718688965, 0.54, 0.45999999999999996], [1.9050071239471436, 0.52, 0.48], [1.8336999416351318, 0.5733333333333334, 0.42666666666666664], [1.1463830471038818, 0.66, 0.33999999999999997], [1.0511069297790527, 0.62, 0.38], [1.1563661098480225, 0.5866666666666667, 0.41333333333333333], [1.5311968326568604, 0.5666666666666667, 0.43333333333333335], [1.3432860374450684, 0.6533333333333333, 0.3466666666666667], [1.4288420677185059, 0.5333333333333333, 0.4666666666666667], [1.7222440242767334, 0.6466666666666666, 0.3533333333333334], [1.6416518688201904, 0.5933333333333334, 0.4066666666666666], [1.9139580726623535, 0.6133333333333333, 0.3866666666666667], [1.7868051528930664, 0.5733333333333334, 0.42666666666666664], [1.147373914718628, 0.4866666666666667, 0.5133333333333333], [1.051367998123169, 0.6133333333333333, 0.3866666666666667], [1.1438648700714111, 0.5133333333333333, 0.4866666666666667], [1.478266954421997, 0.6266666666666667, 0.3733333333333333], [1.4207231998443604, 0.5333333333333333, 0.4666666666666667], [1.4302358627319336, 0.6066666666666667, 0.3933333333333333], [1.7139828205108643, 0.6266666666666667, 0.3733333333333333], [1.6180298328399658, 0.6133333333333333, 0.3866666666666667], [1.918471097946167, 0.5666666666666667, 0.43333333333333335], [1.8050270080566406, 0.5733333333333334, 0.42666666666666664]]
kNN_faces_list_10 = [[1.4868829250335693, 0.6933333333333334, 0.30666666666666664], [6.747021913528442, 0.52, 0.48], [1.2750041484832764, 0.62, 0.38], [1.27783203125, 0.6466666666666666, 0.3533333333333334], [1.3831241130828857, 0.5333333333333333, 0.4666666666666667], [1.6645560264587402, 0.6533333333333333, 0.3466666666666667], [1.5589730739593506, 0.6666666666666666, 0.33333333333333337], [1.651648998260498, 0.5133333333333333, 0.4866666666666667], [1.749406099319458, 0.5133333333333333, 0.4866666666666667], [1.8937709331512451, 0.5333333333333333, 0.4666666666666667], [1.3838610649108887, 0.5733333333333334, 0.42666666666666664], [1.8910551071166992, 0.5266666666666666, 0.4733333333333334], [1.7010090351104736, 0.6133333333333333, 0.3866666666666667], [2.32066011428833, 0.5133333333333333, 0.4866666666666667], [1.92122483253479, 0.52, 0.48], [2.6046841144561768, 0.5133333333333333, 0.4866666666666667], [2.9423789978027344, 0.5466666666666666, 0.45333333333333337], [2.3674910068511963, 0.5133333333333333, 0.4866666666666667], [2.4770069122314453, 0.52, 0.48], [2.598707914352417, 0.5333333333333333, 0.4666666666666667], [1.1542818546295166, 0.5333333333333333, 0.4666666666666667], [1.0457141399383545, 0.5133333333333333, 0.4866666666666667], [1.1659860610961914, 0.52, 0.48], [1.4754910469055176, 0.5133333333333333, 0.4866666666666667], [1.3480730056762695, 0.5133333333333333, 0.4866666666666667], [1.6413400173187256, 0.6, 0.4], [1.521239995956421, 0.56, 0.43999999999999995], [1.618016004562378, 0.52, 0.48], [1.705009937286377, 0.5266666666666666, 0.4733333333333334], [1.7970130443572998, 0.5266666666666666, 0.4733333333333334], [1.1508958339691162, 0.5333333333333333, 0.4666666666666667], [1.055229902267456, 0.52, 0.48], [1.154878854751587, 0.5133333333333333, 0.4866666666666667], [1.4552428722381592, 0.72, 0.28], [1.341465950012207, 0.5333333333333333, 0.4666666666666667], [1.6440889835357666, 0.5933333333333334, 0.4066666666666666], [1.5149810314178467, 0.5733333333333334, 0.42666666666666664], [1.6295011043548584, 0.5466666666666666, 0.45333333333333337], [1.7221670150756836, 0.5466666666666666, 0.45333333333333337], [1.8007349967956543, 0.5266666666666666, 0.4733333333333334], [1.153264045715332, 0.5533333333333333, 0.44666666666666666], [1.0501279830932617, 0.6466666666666666, 0.3533333333333334], [1.1644539833068848, 0.74, 0.26], [1.4803440570831299, 0.54, 0.45999999999999996], [1.3352560997009277, 0.58, 0.42000000000000004], [1.6269619464874268, 0.6266666666666667, 0.3733333333333333], [1.5173649787902832, 0.56, 0.43999999999999995], [1.6183769702911377, 0.5866666666666667, 0.41333333333333333], [1.7144701480865479, 0.5266666666666666, 0.4733333333333334], [1.8055360317230225, 0.5266666666666666, 0.4733333333333334]]

data_size = [10, 20, 30, 40, 50, 60, 70 , 80, 90, 100]
list_digits = [perceptron_digits_list, perceptron_digits_list_5, nb_digits_list, kNN_digits_list, kNN_digits_list_10]
list_faces = [perceptron_faces_list, perceptron_faces_list_5, nb_faces_list, kNN_faces_list, kNN_faces_list_10]
labels = ['Perceptron (Iter = 3)', 'Perceptron (Iter = 5)', 'Naive Bayes', 'KNN (K=5)', 'KNN (K=10)']
def plot_list(lists, dataset, plot_type):
    for idx, tmp in enumerate(lists):
        time = np.asarray([0 for i in range(10)])
        acc = np.asarray([0.0 for i in range(10)])
        acc_std = [[] for i in range(10)]
        for i in range(5):
            for j in range(10):
                time[j] += tmp[i*10+j][0]
                acc[j] += tmp[i*10+j][1]
                acc_std[j].append(tmp[i*10+j][1])
        time = time/5
        acc = acc/5
        acc_std = np.asarray(acc_std)
        acc_std_final = np.std(acc_std, axis=1)
        if plot_type == "time":
            plt.plot(data_size, time, label = labels[idx])
            plt.legend()
            plt.title('Training time vs. The number of data points used for training')
            plt.ylabel('Time (s)')
            plt.xlabel('Training Set Size (%)')
            plt.savefig('time_comparison_' + dataset + '.png', dpi=600)
        elif plot_type == "acc":
            plt.plot(data_size, acc, label = labels[idx])
            plt.legend()
            plt.title('Accuracy (Mean) vs. The number of data points used for training')
            plt.ylabel('Mean of Accuracy')
            plt.xlabel('Training Set Size (%)')
            plt.savefig('mean_acc_comparison_' + dataset + '.png', dpi=600)
        elif plot_type == "acc_std":
            plt.plot(data_size, acc_std_final, label = labels[idx])
            plt.legend()
            plt.title('Accuracy (SD) vs. The number of data points used for training')
            plt.ylabel('Standard Deviation (SD) of Accuracy')
            plt.xlabel('Training Set Size (%)')
            plt.savefig('std_acc_comparison_' + dataset + '.png', dpi=600)

for data in ['digits', 'faces']:
    for dtype in ['time', 'acc', 'acc_std']:
        if data == 'digits':
            plot_list(list_digits, data, dtype)
            plt.clf()
        else:
            plot_list(list_faces, data, dtype)
            plt.clf()
