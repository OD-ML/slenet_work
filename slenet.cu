#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "mnist.h"

float c1_preact[6][24][24]; 
float c1_output[6][24][24]; 
float s1_preact[6][6][6]; 
float s1_output[6][6][6]; 
float f1_preact[10];
float f1_output[10]; 

float c1_bias[6] = {-0.295779, 0.430410, 0.389516, -0.437929, 0.249217, -0.041092, };
float c1_weight[6][25] = {{0.021796, -0.253539, -0.264633, -0.390335, 0.164373, -0.003884, -0.339657, 0.155349, -0.066933, -0.109127, -0.245242, 0.023972, -0.099835, -0.406280, -0.404768, -0.063032, -0.176222, 0.390005, -0.041021, 0.504991, 0.411280, 0.490982, -0.211766, 0.357036, 0.089467, },
{0.462476, -0.390946, 0.353110, 0.051559, -0.291983, 0.032132, 0.267439, -0.133041, -0.042471, 0.009999, -0.369408, 0.172607, -0.354324, -0.121653, -0.264116, 0.124041, -0.445006, 0.147228, 0.044242, -0.334759, -0.394303, 0.404403, -0.443299, -0.057149, 0.400614, },
{-0.120172, -0.363410, 0.140784, 0.426477, 0.527300, 0.055575, 0.433228, 0.274346, -0.415725, -0.284644, -0.352777, 0.220872, -0.017872, 0.166825, -0.156702, -0.014411, -0.182755, -0.025224, 0.451383, 0.104624, -0.390085, -0.433460, -0.244068, 0.163247, -0.256354, },
{0.142145, -0.241728, 0.194633, 0.014756, -0.406083, -0.287899, 0.121829, 0.131957, -0.450947, -0.004920, -0.204048, -0.545422, -0.246691, -0.236730, -0.535327, -0.048552, -0.585163, -0.081254, -0.357632, -0.236740, -0.571868, -0.233020, 0.038282, -0.513554, -0.434952, },
{-0.402966, -0.178654, 0.025999, -0.140377, 0.231947, -0.136444, 0.322751, 0.161807, 0.376979, 0.411809, 0.427200, 0.131421, 0.258028, 0.483045, -0.300380, 0.568240, 0.520461, 0.130053, -0.169407, -0.456362, -0.350305, -0.144329, 0.111549, -0.281440, 0.088230, },
{0.472607, 0.106702, 0.359270, 0.373885, -0.261109, 0.529240, -0.129713, 0.451533, -0.149899, 0.526446, -0.323354, 0.578022, 0.112618, 0.486861, 0.479792, -0.182139, -0.119122, -0.015675, -0.267098, 0.122984, -0.297723, 0.347792, 0.316906, -0.019568, 0.455453, },
};

float s2_bias[1] = {0.827946};  
float s2_weight[1][16] = {{-0.806280, -3.872459, -2.141078, -6.543421, -2.135209, -5.067534, -2.430283, -6.847790, 2.860572, -2.327538, 0.649373, -1.243741, 7.845700, 5.709686, 5.967687, 5.219587, },
};


float f3_bias[10] = {-4.136192, 1.261294, -3.176764, -4.166682, -4.117179, -1.692405, -3.250649, -2.580800, -11.371058, -6.880490, };
float f3_weight[10][216] = {{0.351526, 0.311490, -0.498288, 0.119535, -0.681564, 0.255129, 0.480886, -0.492643, -1.281708, -1.340271, -1.966150, -1.082251, -0.488600, -1.450043, -1.267618, 0.749348, -1.384504, -2.280142, -1.568977, -2.571040, -0.077106, 0.271615, -0.975582, -0.953359, 0.022398, 0.314329, -0.612719, 0.926436, -0.227751, -0.078307, 0.931316, 0.901250, 1.299293, 0.869457, 0.921717, 0.423475, 0.150990, 0.432035, -0.273189, -0.170283, -0.600728, 0.180319, 0.191720, -0.891604, -0.404788, -0.035962, 0.006960, -0.254663, -0.110516, 0.245278, -0.891609, -0.432604, -0.910933, 0.161980, 0.316577, -0.467119, -1.544833, -1.896033, 1.179531, 0.034170, 0.232885, 1.607719, -0.648748, -0.462456, 0.485259, 0.479415, 1.152267, 1.451969, 1.869130, 1.526582, -0.352413, 0.384313, 0.123432, -0.254515, -0.452569, -0.480945, 0.038267, -0.566631, 0.207059, -0.048547, 0.499613, 0.388086, 0.222929, 0.586097, 0.116915, 0.495215, -0.987635, -2.543056, 0.246004, 1.523358, -0.007175, -0.415974, -0.980118, -2.122471, 0.091124, 0.934185, -0.230031, 1.125950, 1.472628, -0.418642, -0.469083, 0.218658, -0.225766, 0.126642, 1.289255, -0.801874, -1.186997, 0.691511, -0.066857, -0.455275, -0.864519, -1.566909, -2.016968, -0.362027, -1.450920, -2.145304, -0.248081, 1.607327, -0.527347, -1.465685, -0.668917, -0.856998, -1.089551, 1.483309, 1.500693, 0.316248, 1.121368, 2.691165, -3.276138, -2.794873, 2.158591, 1.882994, 0.663664, 4.399794, -2.498104, 0.206852, 0.658676, 0.629444, 1.472383, 0.444734, -1.161019, -1.773016, -1.572188, 0.321799, 0.004689, -0.950659, -0.336590, 1.122141, 0.843947, 1.393509, -0.005872, 0.329899, -0.920808, 0.332284, -1.017754, -0.925537, 0.611789, 0.689382, -1.177078, -3.361110, -3.442491, -1.786212, -0.446117, -1.117085, -1.961925, -1.660484, -0.835836, -0.708459, -0.887261, -1.056308, 0.146323, 0.695095, -0.016069, 0.057834, -0.150198, -1.871884, -0.296581, 0.340804, 0.645604, 0.180815, 0.293250, 0.252584, 0.101424, 0.038066, -1.041792, 0.062905, 0.429716, 0.246763, 0.071984, 1.030795, 0.405589, 1.316891, 0.099288, 0.161572, -0.628917, -1.235220, 0.339577, 0.339029, -0.912725, -2.196772, 0.215959, -0.054501, -1.405472, -0.721270, -0.762793, -1.269405, 1.265924, 0.707909, -1.608350, -0.869309, 0.352279, -0.120468, 0.897106, 0.554869, 0.057806, -0.008923, },
{1.072354, 0.368339, -1.401245, -1.984874, -0.154874, 0.688627, 1.252123, -0.170966, -2.428857, -3.910559, -0.865141, 0.669313, 0.903312, 0.585990, -0.680883, -2.790664, -1.126961, 0.799640, 1.482437, 0.040566, -0.812769, -0.477971, 0.341852, 0.863961, 0.605193, 0.013792, 0.662186, -1.722951, 0.585745, 0.924885, 1.817204, 0.060868, 1.821460, -0.918883, -0.057204, 0.797977, 0.452504, 0.496524, 0.344424, 0.016676, -0.094910, 0.523870, 0.278682, -1.489352, -1.921500, -1.428460, -0.564729, 0.523013, 0.165761, 0.126036, 1.813802, 0.663094, -0.318928, 0.770893, 0.669484, -1.068829, 0.739137, -1.242047, -0.439704, 0.559931, 0.865418, -0.137752, 2.103791, -0.997581, 0.782502, 0.366917, 0.823618, 0.888054, 0.286032, -0.361565, -0.430689, 0.281494, 1.016365, 0.160857, -0.855399, -1.359904, -1.563102, -0.028148, 0.433039, -0.098167, 0.425801, 0.482635, -1.104380, -0.527352, 0.585520, 0.331061, -0.891147, 0.136314, -1.160889, 0.565851, 0.551063, -0.376435, -1.301232, 0.362196, -0.400408, 0.630205, 0.791806, 0.054025, -0.342187, 0.930735, 0.411747, -1.107044, 0.491877, 0.019306, -0.101507, -0.070900, 0.602084, 1.044927, 1.431638, -0.186909, -1.135282, -0.511566, 0.171745, 1.461206, -0.232738, -1.303806, -0.149070, -1.895074, -0.602412, -0.266732, 0.098938, -1.655942, 0.386227, 0.933887, -1.909085, 0.372110, 0.096381, -4.576565, 3.424240, 0.455927, -2.730772, 0.495288, -0.962904, -3.011557, 3.950312, -3.998309, -1.283694, 0.121601, -0.364463, 0.343661, -0.329431, -1.021758, 0.770604, 0.189354, 0.332567, -1.434894, -0.826553, -0.353440, 0.025853, 0.213315, -0.121208, -0.273903, -2.700224, -1.330576, -1.555105, -0.189926, 0.745070, 1.035755, -0.314341, -2.745244, -1.638248, -0.129167, 0.975488, -0.669019, -1.211773, -1.447167, 0.393014, 0.894740, 1.373143, 1.001170, 0.112002, -0.675656, 0.418169, -0.494772, -0.095957, 0.099470, -0.360814, -0.891608, 0.262977, 0.920783, 1.389582, 0.287218, -0.372921, 1.515279, 0.964876, 0.336841, 0.641328, 0.264885, -1.132696, -0.681157, -1.149430, -0.574936, 0.626039, 1.129750, -0.087481, 2.067984, -0.583266, 0.990982, 1.091312, 1.595145, -0.956694, -0.864649, -1.045922, 0.756581, 1.174176, 0.200113, 0.244772, -0.676887, 1.685088, -0.092100, 0.609645, 0.356144, 0.205162, 1.691932, 0.513849, 0.718396, },
{-0.691959, -0.319490, -0.126002, -0.501013, -0.875302, -1.076519, -0.957201, 1.113194, 1.674168, 0.329545, -1.761311, -1.159295, 1.613682, 2.941732, 2.173943, -0.021000, -1.825896, -0.170094, 0.173728, -0.361445, -1.320421, -0.518738, 1.295719, 2.961711, -0.759120, -1.930551, 0.547168, 0.954941, 2.664016, 1.866902, 1.253423, 1.417768, 1.667163, 1.042230, -0.100517, 0.300578, -0.742821, -0.435868, 0.064552, -0.705748, -1.213497, -0.001804, -0.359023, 0.298137, -0.462290, -0.185735, -0.328315, -0.338046, 0.657853, 2.652926, 0.959392, -2.879065, -1.569085, -0.166893, -1.256025, 1.924442, -0.765449, -0.368007, 0.810416, 1.628253, -0.657158, 0.502192, -0.500995, 0.748136, 0.398722, 0.901416, -0.039363, 1.347513, 1.276203, -1.614992, 0.438525, 0.500054, 0.113522, -0.032455, -0.627655, 0.008442, 1.360402, -1.020135, -0.221365, -0.158344, 0.732279, 0.599728, 0.105030, -0.698781, -0.691555, 0.842164, -0.142675, -1.220035, -1.879403, -1.355728, 0.951171, 1.169849, -0.774057, -0.290896, -1.583322, 0.386187, 0.427529, -0.284459, 0.255392, -0.747500, 1.408733, 0.321606, -0.201709, -0.559591, -0.594734, -0.110160, -1.030297, -1.282099, 0.072950, 1.014169, 2.120188, -0.186486, -2.108402, -0.302105, -0.778032, 0.225517, 1.845844, 1.028797, -0.750887, -1.268448, -1.074318, -2.053718, -3.085705, -0.494818, -0.322282, 0.149053, -3.596669, -4.111627, -1.300500, 0.565006, -0.487223, 1.346992, -0.219845, 1.634139, 0.398809, 0.230929, 0.485752, 0.666853, -0.158405, -0.415601, -2.933488, -1.255654, 1.545914, 0.507034, -0.493720, 0.380871, 0.083095, 0.623031, -0.399877, -0.657982, -0.623307, -0.873340, 0.656684, 0.734791, -1.335358, -2.101094, 0.831576, 2.100440, 2.293340, -0.346170, -3.333020, -0.954265, 2.421231, 2.408652, 0.758019, 0.139671, -0.483363, 3.165620, -0.050275, 0.177964, 0.436090, -0.539363, 2.990378, 2.356659, -1.532563, -0.977257, -0.316307, -0.759580, -0.439312, -0.550233, -0.031386, -1.135577, 0.329767, -1.219383, -1.346268, -0.968223, -1.073834, -1.273577, -0.268350, 0.434507, -0.372449, -0.663014, 0.776042, 2.483770, 0.461209, 0.288777, -0.179103, -1.031428, -0.596650, -0.469062, 0.658128, 0.681416, -1.432260, -0.364713, -1.425495, -2.454195, -2.924943, 0.629133, 1.855964, 1.667543, -0.539052, -0.663570, -0.630038, 1.244874, 2.319129, 0.256111, },
{0.028224, 0.731431, 0.475040, 0.247407, -0.142814, -0.925812, 0.750290, 2.298835, 2.077607, -1.911288, -3.412653, -2.430810, 2.430562, 2.092515, 0.864522, -1.024722, 1.909442, -0.423104, 1.032791, 0.417014, 2.278990, 1.731266, -0.399394, -2.393260, 2.269246, 1.150298, 1.776157, 0.136200, -2.514576, -2.467292, 0.997515, -0.738169, 1.590271, 1.668116, -0.269244, -1.854345, 0.077514, -0.302778, -0.874067, -0.744745, -0.850755, -0.361762, 0.871575, 0.953886, 0.648477, -0.290329, -3.458227, -1.040640, 1.876841, 1.188782, 1.384314, 0.455497, 1.613461, -1.018162, -2.163025, -1.468017, 1.862550, 2.644363, -1.452993, -0.744264, 0.758142, -0.570616, 0.252323, 0.079597, -1.685202, -1.602713, -0.509896, 0.338261, 0.804202, 1.327994, 1.764287, -0.834753, -0.390510, 0.406986, 0.207928, -0.178557, 0.281699, -1.340411, 0.203804, 0.709745, 1.120675, -0.340546, -1.934595, -1.689729, -0.356253, -0.715572, -0.026291, 0.528598, 1.438491, -0.804467, 1.230794, -0.282758, 1.144114, 2.084298, 1.288977, -1.594794, 1.423724, 0.973438, 1.766188, 0.130267, -0.855660, -0.882746, 0.249111, 0.849037, -0.265883, 0.100048, -0.621917, -0.636140, -0.310966, 1.032739, 2.441257, -0.337551, -0.938824, -0.767724, 0.815726, 0.448318, 0.772167, -1.037305, -0.715248, -2.069259, -0.343856, -3.506028, -0.174516, 0.326002, 0.566527, -1.082228, -3.233820, -2.385847, -0.063642, -1.174212, -1.363371, -2.330489, -2.139360, -3.038171, -2.776845, 0.576879, 0.286743, -0.373866, -0.018202, 0.476584, -0.836212, -0.014623, -0.136377, -1.360772, 0.327430, -0.113351, 0.137505, 0.965799, -0.861879, -0.724133, -0.401539, 0.766025, 1.169944, -0.366683, -3.283610, -2.789598, 0.032294, 4.502056, 4.143749, 0.586006, 0.955034, 0.244452, 0.052637, 0.166199, 0.681101, 1.200757, 1.281066, -2.031965, -0.776959, 0.801955, 0.602662, -0.169299, -1.528645, -2.665930, 0.638968, 1.680237, 3.603701, 2.158989, -0.014057, -0.416837, 0.440411, -0.500588, -0.545031, -0.002605, -0.765310, -0.513167, -0.831632, -1.550555, -0.707097, 0.106397, -0.745175, -1.044604, 0.019416, 0.010024, -2.017556, -2.630480, -1.053369, -1.324165, 1.747035, 2.872766, -1.631575, 0.505731, 1.166912, 1.544383, 1.555105, 2.763806, 1.979878, 1.498059, -0.747289, -1.867625, -0.862028, 1.770981, 0.726319, -0.243810, -1.504762, -0.627387, },
{0.605470, 1.008720, 0.954329, -0.422790, 1.041441, 1.132543, 0.820367, -1.078988, -3.354347, -3.009795, -2.115187, -0.110192, -0.170134, -1.640482, -0.756629, -0.147752, -0.687044, 0.966370, 0.639470, 0.431753, 0.897111, 0.845856, 1.846606, 0.674693, 0.355025, 1.553313, 0.005962, 1.379894, 0.214096, -0.235915, 0.264016, -0.754259, -1.066369, 1.306731, -0.285051, 0.612718, 0.788979, -0.047261, -0.978383, -0.008824, -0.624869, -0.264651, 0.063259, -1.707262, -2.737840, -3.459494, -1.034861, 0.211471, -0.504852, -0.771915, -1.427841, -2.282481, 0.023153, 0.558383, 1.146996, 2.632204, -0.159290, 2.404744, 1.527529, 0.522000, 1.872798, 1.013299, -0.699093, 0.122708, -0.545191, 0.166142, -0.594660, 0.222997, -0.401758, -1.361758, -0.476764, 0.956975, 0.287237, -0.609027, -0.621629, -0.790200, -0.870769, 0.292238, 0.248819, -1.166814, -1.941471, -2.895185, -1.896726, -0.817296, 0.345828, 0.316599, 0.930019, -0.018792, -1.040021, -0.314016, 0.350711, 0.852039, 0.883653, 0.863334, 1.229111, -0.480921, 0.838265, -0.984143, -0.283884, -2.004382, -0.283403, 0.226205, 0.536110, -0.535103, -2.309651, -0.404985, -1.089340, 0.223954, 0.962934, -1.445163, -3.189492, -3.195721, -2.166667, -0.600257, -0.703800, -1.860576, -4.334344, -3.321896, 0.407779, 2.153920, -0.752714, 0.038007, -0.418196, -0.634980, -0.910577, 0.012312, 0.830287, 2.987901, 0.024213, 2.076304, 0.965968, -1.190434, 2.178090, -1.198371, 1.570176, 1.480973, -2.032087, -0.701313, -0.512109, -0.114229, 2.044681, 2.550766, 1.348469, 1.097159, 0.077612, -0.481913, 1.130050, 0.252927, 0.708488, 0.846604, -0.465855, -1.213072, -2.052148, -3.733171, -1.650295, -0.660979, 0.288975, 0.129947, -0.875093, -0.139181, -1.279548, 0.493599, 0.043572, -0.229048, 0.239556, 0.913621, 1.937669, 1.101201, 0.210626, 0.143291, -1.054990, -2.452760, -0.641719, 1.312720, 0.223956, 0.252463, -0.567263, -0.593141, -1.149197, 0.220914, 0.078692, 0.049990, -0.487459, -1.257367, -0.051236, 1.149742, 0.586681, 1.363646, 1.866550, -0.779017, -0.017441, -0.423156, 0.834019, -0.638466, 0.860806, -2.188282, -1.552213, -1.127233, 0.322562, -1.572704, 1.958601, -1.034905, -0.231282, 0.791806, 0.679728, -0.435758, -0.546035, -1.277807, 0.264984, 0.237592, 0.997363, 0.177945, -0.483345, 0.208856, 1.371936, 0.266403, },
{-0.696144, -1.196357, -0.462394, 0.577340, 0.012268, -1.318769, -2.461449, -2.666761, -0.940403, 2.343443, 4.631914, 2.863719, -1.904642, -0.023214, 0.360693, 1.294167, 1.156900, 3.301312, 1.280124, 2.522640, 2.427920, -0.059205, -1.692239, -1.389722, 0.947089, 1.768922, 1.552398, -0.585285, -1.140386, -1.472612, -1.570746, 0.627772, 0.849177, 0.350142, 0.137553, -0.322082, -0.030825, -0.134421, -0.000502, -0.095808, -0.386331, -0.186376, -1.303936, -2.420095, -1.040521, 1.969091, 3.302097, 1.498935, -1.647167, -1.656898, 0.449564, 0.626204, -0.177588, 1.608818, 0.548230, 0.972414, -0.094203, -1.870300, -2.280556, -1.405037, 0.299774, -0.148465, 0.727995, -1.495048, -1.591736, 0.135984, -1.456758, -1.271563, 0.111287, 0.518135, 2.217469, -0.033818, -0.360073, -0.885658, -0.665973, -0.544420, -1.101023, -0.591615, -1.012255, -0.417808, -0.790774, -0.073430, 1.510153, 1.988166, -0.622541, 0.454845, 0.161600, -0.409001, 0.567755, -1.239707, -0.302708, 1.188033, 2.390775, 0.150514, -0.326291, -0.266247, -0.167798, -0.681845, 1.696693, 0.200086, -0.665708, 0.709463, 0.086825, -0.300355, -0.775442, -0.768118, -0.810333, -0.285581, -0.903520, -1.684449, -1.684980, 0.163612, 0.097734, -0.435042, -0.756737, 0.916703, -0.518343, 0.363535, 0.872383, 2.131145, 1.322892, 1.445104, -0.371017, -0.870410, -4.638152, 1.001428, -1.593032, -0.545950, -0.730349, -2.541779, -2.324286, -2.391434, -0.328845, -3.026253, -2.882171, -2.637759, 0.778667, 0.757154, -1.263170, -0.783553, -1.217852, -1.987849, 0.380539, 0.164857, -0.566466, -0.556254, -0.764206, 0.090393, -0.644545, -1.717447, -1.176613, -1.992977, -2.139895, 1.718997, 4.146919, 2.972562, -1.004348, -3.013973, -1.819408, 0.414872, 1.482198, 2.741585, -0.845975, 0.003225, -0.477130, 0.337183, -1.780047, -0.781111, 0.139196, 1.124832, 1.587840, -0.207457, 0.004565, -0.397677, -1.173871, -0.324956, 2.181667, 0.886929, -0.727547, 0.137354, -0.537158, -0.563190, -0.817926, -0.349328, -0.209062, -0.730801, -1.281162, -1.671107, 0.004682, 0.055675, -1.174977, -1.416788, -0.561563, -0.613994, 1.107445, 3.186925, 1.782529, 0.589655, 0.032335, 3.120431, 1.267315, 3.003472, 1.706056, 0.769143, -0.690062, 0.978426, 1.427597, 1.202568, -0.285217, -0.556879, -0.751572, -0.630459, 0.678696, 0.535428, -1.713185, -1.275920, },
{0.152315, -0.765125, -0.751097, -0.934556, -0.828813, -0.360306, 0.553876, -1.715084, -1.507327, 0.238487, 2.936196, 0.171038, -0.453175, -2.462998, -1.797690, 1.452598, 2.208429, -1.393930, -1.170185, -3.140893, -1.269646, 0.142224, -1.178711, 0.184734, -0.379825, -1.661109, -1.783792, -0.136366, 0.680987, 0.705742, -0.610667, -0.642345, 2.008017, 1.448722, 1.486468, -0.174222, 0.053077, 0.057984, 0.098309, 0.063227, 0.511289, 0.596257, -0.175441, -0.699661, -1.050908, 0.796928, 1.716150, 0.038229, -0.196895, -0.708503, -0.074440, 1.416073, 1.032223, -0.242982, -0.146953, -1.486912, 0.032964, -0.886101, -0.021202, 0.452136, 0.740648, -0.089782, 1.320532, 1.728403, 1.096083, 0.021652, -0.514822, 0.107807, -0.084635, 1.582152, 0.410119, 0.101734, 0.479814, -0.304566, -0.499381, -0.988104, -0.644483, 0.202347, -0.175356, -1.139356, -1.922658, -2.457584, -0.756926, -0.347628, -0.050111, -0.861331, -1.630686, -0.272172, 1.494301, 0.593950, 0.098407, -0.035254, -0.485196, -0.324972, -0.200334, -0.022537, -0.239523, -0.493404, -0.038590, 0.721413, 0.069640, -0.356777, 0.298095, -0.714244, 0.481530, 0.390630, 0.768154, 0.354319, 0.814551, -0.500932, -1.042649, 0.753218, 1.810158, 0.855884, -0.243296, -0.888070, -2.149411, -1.668504, -0.348004, -0.490439, 0.192386, -0.437590, -2.646943, -4.076700, -1.950660, -1.213432, 1.292691, 2.234244, -0.728314, -0.311757, 0.004605, 0.809414, -0.933965, 4.031323, 1.072258, 0.558371, 0.365051, 0.223374, -1.044929, -0.579311, -0.585072, -0.728306, -0.239855, 0.495314, 0.534240, 0.467650, -0.707637, -0.919751, -0.717726, -0.592283, -0.360222, -0.062120, -0.469600, -0.227160, 1.538600, 1.377977, -0.359240, -0.935132, 0.207331, 1.089410, 3.445698, 0.979894, -0.691586, -1.640150, -0.714641, 1.529299, 0.496909, -0.973835, -0.623223, -1.958644, -1.392360, 0.807418, 1.336986, -0.049339, 0.271819, 0.077745, -0.620963, 1.114028, 1.931875, 0.191675, 0.118725, 0.075329, 0.176480, 0.313298, 0.170722, -0.035340, 0.718942, -0.976279, -1.643846, -1.809573, -0.426578, -0.676316, -0.454686, -0.373467, -2.138366, 0.126349, 1.922013, 1.565262, -0.369933, -1.934996, -2.577240, 1.793829, 0.798392, 0.262219, 0.464289, -1.430542, -1.801765, 0.516793, -0.083053, -0.174132, 0.139066, 0.460361, 0.274277, -0.090929, -0.303901, -0.013966, },
{0.574065, 0.087913, -0.101225, -0.425271, 0.837332, 0.443861, 0.829129, 1.649770, 1.595022, 1.275656, 1.208999, 0.306782, 1.609037, 2.203841, 0.832572, 0.196236, 0.687585, -0.140099, 1.119247, 0.917588, -1.346617, 0.448468, 0.853814, -0.761280, 0.688738, -0.607088, -2.214644, -0.748777, -0.123899, -0.393272, -0.130794, -3.265965, -2.664813, -1.615597, -1.341889, 0.006544, -0.130847, -0.088353, -0.180492, 0.562549, 0.575968, 0.773054, 0.057666, -0.987964, 0.160948, 1.771033, 0.140249, -0.430556, 1.742646, 1.219949, 1.044510, -0.459169, 1.725265, -0.157969, 0.212724, 0.747507, -3.143275, -1.688458, 0.138393, -0.066797, -0.273624, -1.673753, -1.791322, -0.278081, -0.308548, -0.592532, -0.036351, -0.317496, -1.176857, -1.091496, -1.852126, -0.262260, 0.225579, 0.148371, -0.224067, -0.256520, -0.785286, 0.648806, 1.282037, 0.524275, 1.131838, 1.522663, 0.081340, -0.849995, 1.464951, 0.415561, 0.547651, -1.214204, -0.770125, -0.331421, 0.347348, -0.886304, -1.188123, -0.314620, 0.258324, -0.966734, -0.145145, -0.450169, -1.788401, -0.985112, -1.140127, -0.648513, 0.489472, -0.470473, -1.734630, -0.641057, 0.080053, -0.003697, 0.685148, -0.051665, -0.236035, 0.663310, 0.154144, 0.704001, 0.230949, 2.886303, 1.934678, 0.203823, -1.470700, -1.130880, -0.171957, -1.121534, -0.097719, 1.766702, -0.120937, -1.068842, -0.986347, 0.332667, -4.731073, 0.397900, 2.130679, 0.326992, -2.078055, -3.046449, 0.330215, -0.637420, -0.818264, -1.098654, 0.227594, -1.287947, 0.766962, 0.168925, -2.022322, -0.330416, 0.222381, 0.101943, -0.903221, -0.482411, -0.297447, 0.206530, -0.645640, -0.845798, 0.234526, 1.614704, 1.224527, -0.291447, -0.381585, 0.358654, -0.627844, -2.057576, -0.084345, 0.675204, 0.725045, 4.221374, 0.921818, -1.301100, 1.063276, -0.241603, -0.414594, 0.559449, -1.050303, -1.688972, -0.476660, 0.628729, 1.667690, -0.067479, -1.875710, -1.654460, -1.064784, 0.546918, 0.408417, -0.421641, -0.297248, 0.041914, 0.372104, 0.628411, 1.316536, 1.628041, 1.474132, 0.602927, 0.593461, 0.460339, 1.150599, 2.164397, 1.562372, 0.326256, -0.488896, -0.140523, -0.045086, -0.054528, -2.512095, -1.983680, 0.940187, 0.054489, 0.310126, -0.674056, -0.772499, -1.438876, -0.919730, -0.455367, 0.895496, -0.803045, -1.692005, -0.201039, -0.416314, -0.165527, },
{0.134199, 0.086146, 0.951013, 1.546682, -0.228825, 0.657228, -0.817823, -2.899863, -0.006010, 1.219655, -0.654646, -1.133906, -1.049594, -1.326216, -0.021972, -0.409191, 0.079562, -0.174925, 0.060131, -0.395260, -0.718243, 0.548309, 1.000185, -0.209588, -2.670066, -4.040078, 1.123819, -0.331347, -3.162083, -0.897409, -0.086779, -1.294813, 1.673631, 0.362336, -1.709957, -0.544266, -0.114851, 0.646117, -0.156913, -0.269812, -0.359048, -0.240598, -0.056497, -1.430671, 0.331621, -0.185001, 0.506179, 0.156263, -0.215623, -1.748398, -1.163215, 0.792030, 1.249370, 0.275016, 0.980882, -1.546848, 0.849806, 0.778316, 1.835576, -0.443658, -1.345628, -0.254083, -0.307223, -1.476557, -2.464857, -0.906314, 0.472844, 0.153223, 0.916816, 1.005717, 0.851677, 0.517156, 0.398625, -0.678162, 0.105231, 0.271946, 0.257889, -0.193391, -0.185805, -0.619955, -0.152400, -0.269105, 0.030182, -0.756154, 0.197620, 1.438872, 1.119218, 0.945686, 1.712977, 1.455088, 0.318649, 1.536823, -1.772251, 0.846019, 1.444420, 0.500221, -1.166500, -0.747753, -2.232754, 0.410973, -0.747600, -0.122100, 0.279852, -0.233738, 1.064610, 0.144592, 0.991708, 0.189145, 0.294694, -1.789718, -2.850063, -0.826650, -1.277863, -1.129428, -0.727659, -0.662391, -1.144246, -0.739091, -0.233197, 0.225324, 0.779543, 2.830250, 3.156541, -1.894109, 1.581901, 3.498926, -0.827843, -0.369051, 2.749464, -0.556152, -0.895778, -1.573576, 0.563548, 4.213789, 0.377439, -1.091892, -1.965009, 0.770312, 1.126254, 2.002281, 2.154553, 0.459983, -0.197655, 0.116272, -0.031684, 0.606186, 1.142404, 0.583559, 1.521510, 1.287277, 0.569019, 0.469541, 1.697982, 0.820891, 0.441047, -0.927124, -0.068266, -1.768230, -1.092471, 0.745175, 1.603513, 0.724154, 1.088344, 0.627051, -1.384045, 0.391496, 2.744067, 1.380161, 0.272975, 1.399887, -0.475394, -0.978054, -2.643478, -2.071975, -1.569098, -4.466246, -0.691999, 2.273736, 0.512966, -0.363127, 0.293437, -0.561686, 0.144939, -0.494503, -0.591951, -0.102512, -0.124045, -1.194018, -1.718056, -1.442958, 0.306223, 0.200272, 0.410612, 0.729493, 1.178130, -1.798445, -0.877996, -0.311697, 0.221705, -1.120724, -0.387341, -2.525295, -0.069459, 2.124259, -0.431480, -7.913769, -5.263272, 0.354615, 0.294655, -0.477982, 0.206353, -1.184989, -0.564744, 0.910499, -1.108228, -0.312997, },
{0.108912, 0.801137, 0.089786, 3.061978, 0.131143, 0.528377, 0.016495, -0.192003, 1.402572, 0.728240, -0.515489, -0.170898, -2.549391, -2.933669, 0.362581, -1.394158, -2.118942, -1.363451, 0.251453, -0.608348, 2.283336, -0.042631, -1.672400, -0.421950, 0.627726, 1.107371, 0.543287, 0.262856, -0.258237, -0.484534, -0.549471, -1.574040, -1.983789, -1.908992, -1.596069, 0.110330, 0.766687, 0.806803, 1.662844, -0.124929, 0.212533, 0.357445, 0.133186, 0.471907, 2.308363, 0.320307, 1.420421, -0.314725, -0.129012, -1.086885, 0.026290, 1.991676, -0.800227, -0.161923, -0.624451, -1.410149, 2.696423, 1.348279, 0.333253, -0.423011, 0.045990, 0.936819, 1.426559, -0.556490, -0.683724, 0.163982, -0.729385, 0.123012, -0.636313, -1.728099, -0.437804, 0.508405, 0.207203, 0.482386, -0.815239, 0.556439, -0.667619, 0.229952, -0.343871, 1.010474, 0.427395, 1.489682, 0.957788, -0.500657, -0.010511, -0.128997, -0.058528, 1.574467, -0.330596, -0.136817, -0.526747, 1.745476, 1.758766, 0.456409, -1.792545, -0.843850, -0.049152, 0.538514, 0.220868, -0.819065, 0.179082, 0.067778, 0.769431, 0.189715, -1.370261, -0.753780, 1.453471, 1.515521, 1.029526, 1.110618, -0.955434, -2.103319, 0.189333, 0.322453, -2.659086, -4.978818, -0.049505, 0.973473, -1.905865, -2.640009, -0.105180, 1.389036, -0.307081, -0.508985, -0.835303, -0.311752, 0.523407, 2.166173, 2.621752, 0.930665, -1.002554, -1.352269, 0.046791, -0.572948, 0.256177, -0.301291, -0.949280, -1.240941, -1.116077, -1.410568, -0.110100, -1.378575, -0.546271, 1.063461, 0.472734, 0.297437, 1.072940, 2.277367, 1.364787, -0.652501, 0.236764, 0.648488, 1.103472, 2.550184, 0.254869, -0.194362, 0.299237, 0.144144, -1.256917, 0.083524, -1.394666, -1.584453, -2.355079, -4.239496, 1.481026, 1.628111, -3.564660, -0.691804, -0.433910, -0.086805, 1.625490, 0.531869, -0.505732, 0.013376, 0.998163, 2.191412, 0.004810, -0.898646, 0.815720, 0.909740, 0.154502, 0.901442, -0.074022, 0.264115, -0.568966, -0.078755, 0.489646, -1.829732, -2.264266, 0.205467, 0.231962, 1.188804, -0.011500, -2.126064, -2.155711, -1.202976, 2.237792, 2.269261, -0.301659, 1.478863, 0.868357, -0.553034, -1.619357, -0.711456, 0.188402, 0.435972, 0.544251, -2.172204, -2.189834, 0.307477, 1.074746, -0.409552, -1.660080, -0.987157, -0.202801, 2.111876, },
};


// layer class declaration 
class Layer{
	public: 
	int M, N, O; 
	float *preact, *output; 
	float *bias, *weight; 

	Layer(int M, int N, int O); 
	~Layer(); 

	void setOutput(float *data); 
	void clear(); 
};

Layer::Layer(int M, int N, int O){
	this->M = M; this->N = N; this->O = O; 

	float h_bias[N]; 
	float h_weight[N][M]; 

	preact = NULL; output = NULL; bias = NULL; weight = NULL; 
	
	for(int i=0; i<N; i++){
		//h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX); 
		h_bias[i] = -1.0f; 
		for(int j=0; j<M; j++){
			//h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
			h_weight[i][j] = 1.0f; 
		}
	}

	cudaMalloc(&output, sizeof(float)*O); 
	cudaMalloc(&preact, sizeof(float)*O); 
	
	cudaMalloc(&bias, sizeof(float)*N); 
	cudaMalloc(&weight, sizeof(float)*M*N); 

	cudaMemcpy(bias, h_bias, sizeof(float)*N, cudaMemcpyHostToDevice); 
	cudaMemcpy(weight, h_weight, sizeof(float)*M*N, cudaMemcpyHostToDevice); 
}

Layer::~Layer(){
	cudaFree(preact); 
	cudaFree(output);

	cudaFree(bias); 
	cudaFree(weight); 
}

void Layer::setOutput(float *data){
	cudaMemcpy(output, data, sizeof(float)*O, cudaMemcpyHostToDevice); 
}

void Layer::clear(){
	cudaMemset(preact, 0x00, sizeof(float)*O); 
	cudaMemset(output, 0x00, sizeof(float)*O); 
}


// input and convolution layers initialization 
static Layer l_input = Layer(0, 0, 28*28); 
static Layer l_c1 = Layer(5*5, 6, 6*24*24); 
static Layer l_s1 = Layer(4*4, 1, 6*6*6); 
static Layer l_f1 = Layer(6*6*6, 10, 10); 

/* kernels */ 
__global__ void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5]){
	const int pos = blockIdx.x * blockDim.x + threadIdx.x; 
	const int size = blockDim.x * gridDim.x; 

	const int N = 5*5*6*24*24; 

	for(int n=N * pos / size; n< N * (pos+1) / size; n++){
		int idx = n; 
		const int i1 = ((idx /= 1) % 5); 
		const int i2 = ((idx /= 5) % 5); 
		const int i3 = ((idx /= 5) % 6); 
		const int i4 = ((idx /= 6) % 24); 
		const int i5 = ((idx /= 24) % 24); 
		//if (pos < 38){
		//	if (blockIdx.x == 0) printf("[%d] ", i4+i1 ); 
		//}
		atomicAdd(&preact[i3][i4][i5], weight[i3][i1][i2] * input[i4+i1][i5+i2]); 
	}
}

__global__ void fp_bias_c1(float preact[6][24][24], float bias[6]){
	const int pos = blockIdx.x * blockDim.x + threadIdx.x; 
	const int size = blockDim.x * gridDim.x; 

	const int N = 6*24*24; 
	for(int n = N*pos/size; n < N*(pos+1)/size; n++){
		int idx = n; 
		const int i1 = ((idx /= 1) % 6); 
		const int i2 = ((idx /= 6) % 24); 
		const int i3 = ((idx /= 24) % 24); 

		preact[i1][i2][i3] += bias[i1]; 
	}
}

__global__ void apply_sigmoid_function(float *input, float *output, const int N){
	const int pos = blockIdx.x * blockDim.x + threadIdx.x; 
	const int size = blockDim.x * gridDim.x;
	
	for (int idx = N*pos/size; idx < N*(pos+1)/size; idx++){
			output[idx] = 1/ (1+exp(-input[idx])); 
	}
}

__global__ void fp_preact_s1(float input[6][24][24], float preact[6][6][6], float weight[1][4][4]){
	const int pos = blockIdx.x * blockDim.x + threadIdx.x; 
	const int size = blockDim.x * gridDim.x; 

	const int N = 4*4*6*6*6; 

	for(int n=N * pos / size; n< N * (pos+1) / size; n++){
		int idx = n; 
		const int i1 = ((idx /= 1) % 4); 
		const int i2 = ((idx /= 4) % 4); 
		const int i3 = ((idx /= 4) % 6); 
		const int i4 = ((idx /= 6) % 6); 
		const int i5 = ((idx /= 6) % 6); 
	
		//	if (blockIdx.x == 0) printf("[%d] ", i4+i1 ); 

		atomicAdd(&preact[i3][i4][i5], weight[0][i1][i2] * input[i3][i4*4+i1][i5*4+i2]); 
	}
}

__global__ void fp_bias_s1(float preact[6][6][6], float bias[1]){
	const int pos = blockIdx.x * blockDim.x + threadIdx.x; 
	const int size = blockDim.x * gridDim.x; 

	const int N = 6*6*6; 
	for(int n = N*pos/size; n < N*(pos+1)/size; n++){
		int idx = n; 
		const int i1 = ((idx /= 1) % 6); 
		const int i2 = ((idx /= 6) % 6); 
		const int i3 = ((idx /= 6) % 6); 

		preact[i1][i2][i3] += bias[0]; 
	}
}

__global__ void fp_preact_f1(float input[6][6][6], float preact[10], float weight[10][6][6][6]){
	const int pos = blockIdx.x * blockDim.x + threadIdx.x; 
	const int size = blockDim.x * gridDim.x; 

	const int N = 10*6*6*6; 

	for(int n=N * pos / size; n< N * (pos+1) / size; n++){
		int idx = n; 
		const int i1 = ((idx /= 1) % 10); 
		const int i2 = ((idx /= 10) % 6); 
		const int i3 = ((idx /= 6) % 6); 
		const int i4 = ((idx /= 6) % 6);  
	
		//	if (blockIdx.x == 0) printf("[%d] ", i4+i1 ); 

		atomicAdd(&preact[i1], weight[i1][i2][i3][i4] * input[i2][i3][i4]); 
	}
}

__global__ void fp_bias_f1(float preact[10], float bias[10]){
	const int pos = blockIdx.x * blockDim.x + threadIdx.x; 
	const int size = blockDim.x * gridDim.x; 

	const int N = 10; 
	for(int idx = N*pos/size; idx < N*(pos+1)/size; idx++){ 
		const int i1 = ((idx /= 1) % 10); 

		preact[idx] += bias[idx]; 
	}
}



void printTests_conv(){
	int i,j,k; 
	cudaMemcpy(c1_preact, l_c1.preact, sizeof(float)*l_c1.O, cudaMemcpyDeviceToHost); 
	cudaMemcpy(c1_output, l_c1.output, sizeof(float)*l_c1.O, cudaMemcpyDeviceToHost); 

	for(i=0;i<6;i++){
		for(j=0; j<24; j++){
			for(k=0; k<24; k++){
				printf("%f ", c1_preact[i][j][k]); 
			}printf("\n"); 
			for(k=0; k<24; k++){
				printf("%f ", c1_output[i][j][k]); 
			}printf("\n"); 

		}printf("\n"); 
	}printf("\n"); 
}

void printTests_ss(){
	int i,j,k; 
	cudaMemcpy(s1_preact, l_s1.preact, sizeof(float)*l_s1.O, cudaMemcpyDeviceToHost); 
	cudaMemcpy(s1_output, l_s1.output, sizeof(float)*l_s1.O, cudaMemcpyDeviceToHost); 

	for(i=0;i<6;i++){
		for(j=0; j<6; j++){
			for(k=0; k<6; k++){
				printf("%f ", s1_preact[i][j][k]); 
			}printf("\n"); 
			for(k=0; k<6; k++){
				printf("%f ", s1_output[i][j][k]); 
			}printf("\n"); 

		}printf("\n"); 
	}printf("\n"); 
}

void printTests_fc(){
	int i; 
	cudaMemcpy(f1_preact, l_f1.preact, sizeof(float)*l_f1.O, cudaMemcpyDeviceToHost); 
	cudaMemcpy(f1_output, l_f1.output, sizeof(float)*l_f1.O, cudaMemcpyDeviceToHost); 

	float max = f1_output[0];
	int m_idx = 0; 
	for(i=0;i<10;i++){
				//printf("[%d]%f ", i, f1_preact[i]); 
				//printf("[%d]%f ", i, f1_output[i]); 
				if (f1_output[i] > max){
					max = f1_output[i]; 
					m_idx = i; 
				}
	}
	printf("[%d]%f \n", m_idx, max); 
}

static double forward_pass(double data[28][28]){
	
	float input[28][28]; 
	for(int i=0; i<28; i++){
		for(int j=0; j<28; j++){
			input[i][j] = data[i][j]; 
			//input[i][j] = 1.0f;  
		}
	}

	l_input.clear(); 
	l_c1.clear();
	l_s1.clear();
	l_f1.clear(); 

	//clock_t start, end; 
	//start = clock();
	cudaEvent_t start, end; 
	cudaEventCreate(&start); 
	cudaEventCreate(&end); 
	cudaEventRecord(start); 

	l_input.setOutput((float *)input); 

	// 1. convolution 
	fp_preact_c1<<<64,64>>> ((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight);
	fp_bias_c1<<<64,64>>> ((float (*)[24][24])l_c1.preact, l_c1.bias);
	apply_sigmoid_function<<<64,64>>> (l_c1.preact, l_c1.output, l_c1.O); 
	//printTests_conv(); 

	// 2. subsampling 
	fp_preact_s1<<<64,64>>> ((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight);
	fp_bias_s1<<<64,64>>> ((float (*)[6][6])l_s1.preact, l_s1.bias);
	apply_sigmoid_function<<<64,64>>> (l_s1.preact, l_s1.output, l_s1.O); 
	//printTests_ss(); 

	// 3. full_connection 
	fp_preact_f1<<<64,64>>> ((float (*)[6][6])l_s1.output, l_f1.preact, (float (*)[6][6][6])l_f1.weight);
	fp_bias_f1<<<64,64>>> (l_f1.preact, l_f1.bias);
	apply_sigmoid_function<<<64,64>>> (l_f1.preact, l_f1.output, l_f1.O); 
	//printTests_fc(); 

	//end = clock(); 
	//return ((double) (end - start)) / CLOCKS_PER_SEC;
	cudaEventRecord(end); 
	cudaEventSynchronize(end);
	float ms = 0; 
	cudaEventElapsedTime(&ms, start, end);
	cudaEventDestroy(start); 
	cudaEventDestroy(end); 
	return (double)ms; 
}

double time_taken = 0.0; 
static unsigned int classify(double data[28][28]){
	
	float res[10]; 
	time_taken += forward_pass(data); 
	//printf("time_taken = %f\n", 1000*time_taken); 
	unsigned int max = 0; 
	cudaMemcpy(res, l_f1.output, sizeof(float)*10, cudaMemcpyDeviceToHost); 
	for(int i=0; i<10; i++){
		if (res[max] < res[i]){
			max = i; 
		}
	}
	return max; 
}

int main(){
	int ret; 
	int i; 
	mnist_data *test_set; 
	static unsigned int test_cnt; 

	// load data 
	if(ret = mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", &test_set, &test_cnt) != 0)
		printf("An error occured: %d \n", ret); 
	else
		printf("test_cnt = %d \n", test_cnt);
	
	// copy to device for the trained parameters 
	cudaMemcpy(l_c1.bias, c1_bias, sizeof(float) * 6, cudaMemcpyHostToDevice);
	cudaMemcpy(l_c1.weight, c1_weight, sizeof(float)*6*25, cudaMemcpyHostToDevice); 

	cudaMemcpy(l_s1.bias, s2_bias, sizeof(float) * 1, cudaMemcpyHostToDevice);
	cudaMemcpy(l_s1.weight, s2_weight, sizeof(float)*16, cudaMemcpyHostToDevice); 
						    
	cudaMemcpy(l_f1.bias, f3_bias, sizeof(float) * 10, cudaMemcpyHostToDevice);
	cudaMemcpy(l_f1.weight, f3_weight, sizeof(float)*10*6*6*6, cudaMemcpyHostToDevice); 
	
	// forward pass 
	unsigned int error = 0;
	unsigned int max = 0; 
	float res[10]; 
	for (i=0; i<test_cnt; i++){
		time_taken += forward_pass(test_set[i].data); 
		cudaMemcpy(res, l_f1.output, sizeof(float)*10, cudaMemcpyDeviceToHost); 
		for(int j=0; j<10; j++){
			if (res[max] < res[j])
				max = j; 
		}
		if (max != test_set[i].label) ++error; 
	}
	printf("Error Rate = %f%% (%d out of 10,000)\n", double(error)/double(test_cnt)*100.0, error); 
	printf("Ex time = %f (ms) \n", time_taken);
	return 0; 
}
