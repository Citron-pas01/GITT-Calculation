import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
"""
%Foumular for LiFePO4
% D=4/(pi*t)*(n*V/S)^2*(Es/Et)^2
"""
# the formular for the diffusion coefficient
#         4  [  js   dEeq/dx      ]2
#   D = ---- [ ---- ------------- ]
#        pi  [ cmax  dE/dsqrt(t)  ]

# Assume the lithium concentration to be close to equilibrium  after the relation period
# dEeq/dx is approximated by (E4-E0)/delta(x), delta(x) = -I_p t_p/(F V_acm C_max), Js = I_p/(F A_acm)
# A_acm is the electrochemically active surface area of the cathode

# the above formular turns into:
#         4  [  V_acm     E4 - E0      ]2
#   D = ---- [ ------- -------------   ]
#        pi  [ A_acm t_p  dE/dsqrt(t)  ]

m = 51.45   # mg of the active material
S = 54.1  # mm^2
Mb = 157.757  # g/mol for the LiFePO4
rou = 3.6  # g/cm^3 for the LiFePO4
#r_p = (m*10**(-3)/Mb * (Mb/rou))/(S*10**(-2))   # end with cm unit
t_p = 30*60

r_p = 3.5*10**(-4)   # in cm

data = pd.read_csv("GITT.csv")
data[["day", "time"]] = data.TestTime.str.split("-", expand=True)
data[["hour", "minute", "second"]] = data.time.str.split(":", expand=True)
data = data.drop(['TestTime', "time"], axis=1)
data = data.astype(float)

data["CumulativeTime"] = data["day"]*(24*3600) + data["hour"]*3600 + data["minute"]*60 + data["second"]

data["CumulativeTime"] = data["CumulativeTime"] - data["CumulativeTime"].iloc[0]

data = data.drop(["day", "hour", "minute", "second"], axis=1)
data = data.reindex(["CumulativeTime", "Current/mA", "Voltage/V"], axis=1)

# Since there is a NA row between the Charge and discharge data recording,
# the original data is divided into the charge_data and discharge_data
charge_end_index = data["CumulativeTime"].index[data["CumulativeTime"].isna()].values[0]

charge_data = data.iloc[0:charge_end_index][:]

discharge_data = data.iloc[charge_end_index+1:-1][:]

ax1 = data.plot.scatter(x="CumulativeTime", y="Voltage/V", c="DarkBlue")
plt.show()
#print(data)

df_raw = pd.DataFrame(np.transpose([data["CumulativeTime"], data["Voltage/V"]]), columns=["TestTime", "Potential"])
df_raw.to_csv("GITT_curve.csv")

def last_period_with_0_drop(dataset, end_position):
    global end_idx, i_p
    for idx in range(end_position, 0, -1):
        if dataset["Current/mA"].iloc[idx] != 0:
            i_p = data["Current/mA"].iloc[idx]
            end_idx = idx
            break


def last_period_not_0_drop(dataset, end_position):
    global end_idx, period_num
    for idx in range(end_position, 0, -1):
        if dataset["Current/mA"].iloc[idx] == 0:
            period_num += 1
            end_idx = idx
            break


# to drop the first period, the test is assumed to start with current testing
def first_period_not_0_drop(dataset, data_length):
    global start_idx, period_num
    for idx1 in range(0, data_length):
        if dataset["Current/mA"].iloc[idx1] == 0:
            period_num += 1
            start_idx = idx1
            break

# to drop the last period, the test finished with a rest can keep the last
print(data["Current/mA"].iloc[-1])

output = pd.DataFrame()
for (data_for_clean, label) in [(charge_data, "charge"), (discharge_data, "discharge")]:
    data_length = data_for_clean.shape[0]
    start_idx = 0
    end_idx = data_length
    period_num = 0
    first_period_not_0_drop(data_for_clean, data_length)
    first_n_last_dropped = []
    soc_end = True

    if data_for_clean["Current/mA"].iloc[-1] != 0:
        i_p = data_for_clean["Current/mA"].iloc[-1]
        last_period_not_0_drop(data_for_clean, data_length-1)
        end = end_idx
        soc_end = False
        #last_period_with_0_drop(data,end)

    #print(start_idx, end_idx)
    first_n_last_dropped = data_for_clean.iloc[:][start_idx:end_idx]
    #print(first_n_last_dropped)

    E_0_idx = 0
    E_0 = []
    E_1 = []
    E_2 = []
    E_3 = []
    new_length = first_n_last_dropped.shape[0]
    #print(new_length)
    for n in range(0, new_length-1):
        if first_n_last_dropped["Current/mA"].iloc[n] == 0 and first_n_last_dropped["Current/mA"].iloc[n+1] != 0:
            E_0.append(first_n_last_dropped["Voltage/V"].iloc[n])
            E_1.append(first_n_last_dropped["Voltage/V"].iloc[n+1])
        elif first_n_last_dropped["Current/mA"].iloc[n] != 0 and first_n_last_dropped["Current/mA"].iloc[n+1] == 0:
            E_2.append(first_n_last_dropped["Voltage/V"].iloc[n])
            E_3.append(first_n_last_dropped["Voltage/V"].iloc[n+1])

    E_4 = E_0[1:]
    E_4.append(first_n_last_dropped["Voltage/V"].iloc[-1])

    delta_E21 = np.array(E_2) - np.array(E_1)
    delta_E40 = np.array(E_4) - np.array(E_0)
    delta_E30 = np.array(E_3) - np.array(E_0)
    IR_drop_1 = np.array(E_1) - np.array(E_0)
    IR_drop_2 = np.array(E_2) - np.array(E_3)
    D = 4/(9*np.pi) * np.power(np.divide(delta_E40, delta_E21), 2) * np.power(r_p, 2)/t_p
    D_trail = 4/(9*np.pi) * np.power(np.divide(delta_E40, delta_E30), 2) * np.power(r_p, 2)/t_p

    total_period_num = period_num + len(E_2)
    instant_soc = np.array(range(1, total_period_num+1))
    delta_x = instant_soc/total_period_num

    if soc_end:
        x_axis = delta_x[1:]
    else:
        x_axis = delta_x[1:-1]

    plt.plot(x_axis, D_trail)
    #plt.show()

    cols = pd.MultiIndex.from_tuples([(label, "SOC"),
                                      (label, "D"),
                                      (label, "D_approx")])
    df = pd.DataFrame(np.transpose([x_axis, D, D_trail]), columns=cols)

    output = pd.concat([output, df], axis=1).reset_index(drop=True)
output.to_csv("GITT_updated.csv")





