
import matplotlib.pyplot as plt

import PySpice.Logging.Logging as Logging

from MySpice import MySpice as spice

logger = Logging.setup_logging()
# for i in range(2048):
#     circuit = spice.LoadFile('data_cir/' + str(i) + '.cir')
#     input_data = spice.Init_Data(1000, 0.3)
#     analysis = spice.CreateCVC(circuit, input_data, 100)
#     spice.SaveFile(analysis, "data_set/set_" + str(i) + ".csv")
#     figure1 = plt.figure(1, (10, 5))
#     plt.grid()
#     plt.plot(analysis.input_dummy, analysis.VCurrent)
#     plt.xlabel('Напряжение [В]')
#     plt.ylabel('Сила тока [А]')
#     plt.savefig("data_set/set_" + str(i) + ".png")
#     # plt.show()
#     # plt.close(figure1)
#     plt.clf()

circuit = spice.LoadFile('3_win.cir')
input_data = spice.Init_Data(1000, 0.3, SNR=10**6)
analysis = spice.CreateCVC(circuit, input_data, 100)
spice.SaveFile(analysis, "3_win.csv")
figure1 = plt.figure(1, (10, 5))
plt.grid()
plt.plot(analysis.input_dummy, analysis.VCurrent)
plt.xlabel('Напряжение [В]')
plt.ylabel('Сила тока [А]')
# plt.savefig("data_set/set_" + str(i) + ".png")
plt.show()
# plt.close(figure1)
# plt.clf()