import xspec
import numpy as np

def compute_cstat_with_background():

    xspec.Plot.setRebin(minSig=None, maxBins=None)
    xspec.Plot.xAxis = "channel"
    xspec.Plot.background = True
    xspec.Plot("counts")
    cstat = []
    dof = []
    for N in range(1, xspec.AllData.nSpectra+1):
        
        W_cstat = 0
        SRC_BACKSCAL = xspec.AllData(N).backScale
        BGD_BACKSCAL = xspec.AllData(N).background.backScale

        t_s = xspec.AllData(N).exposure
        t_b = xspec.AllData(N).background.exposure * BGD_BACKSCAL/SRC_BACKSCAL

        data = np.array(xspec.AllData(N).values)*t_s
        background = np.array(xspec.AllData(N).background.values)*t_b
        model = xspec.Plot.model(N)
        channels = xspec.Plot.x(N)
        dof.append(len(data))
        for i in range(0, len(data)):

            t_i = t_s+t_b
            model[i] = np.max([model[i], 1e-5/t_b])

            S = data[i]
            B = background[i]
            M = model[i]/(t_s)

            a = t_i
            b = t_i*M-S-B
            c = -B*M
            d = np.sqrt(b*b - 4.0*a*c)
            if b >= 0:
                f = -2*c/(b + d)
            else:
                f = -(b - d)/(2*a)

            if data[i] < 0:
                print(f"channel=", channels[i],
                      "data=", data[i], "model=", model[i])
            if data[i] == 0:
                W_cstat += t_s*M - B * np.log(t_b/(t_s+t_b))
            if data[i] > 0:
                if B == 0:
                    if M < S/(t_b+t_s):

                        W_cstat += -t_b*M - S * np.log(t_s/(t_b+t_s))

                    else:

                        W_cstat += t_s*M + S * (np.log(S)-np.log(t_s*M)-1)
                else:
                    W_cstat += (t_s*M + (t_s + t_b) * f - S * np.log(t_s * (M + f)) -
                                B * np.log(t_b * f) - S * (1 - np.log(S)) - B * (1-np.log(B)))
        W_cstat *= 2
        cstat.append(W_cstat)
    return cstat,dof
