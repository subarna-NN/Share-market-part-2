import matplotlib.pyplot as plt, numpy as np
def plot_coverage(naive, naive_ci, calib, calib_ci, alphas=("0.65","0.75")):
    x=np.arange(len(alphas)); w=0.34
    nl=[n-c[0] for n,c in zip(naive,naive_ci)]; nh=[c[1]-n for n,c in zip(naive,naive_ci)]
    cl=[v-c[0] for v,c in zip(calib,calib_ci)]; ch=[c[1]-v for v,c in zip(calib,calib_ci)]
    fig,ax=plt.subplots(figsize=(6.4,4.2),dpi=150)
    ax.bar(x-w/2,naive,w,yerr=[nl,nh],capsize=5,color="#E8A33D",label="naive ensemble")
    ax.bar(x+w/2,calib,w,yerr=[cl,ch],capsize=5,color="#2B8C8C",label="conformal-calibrated")
    ax.axhline(95,color="k",ls="--",lw=1.2,label="nominal 95%")
    ax.set_xticks(x); ax.set_xticklabels([f"$\\alpha={a}$" for a in alphas])
    ax.set_ylabel("empirical coverage (%)"); ax.set_xlabel("held-out memory order")
    ax.set_ylim(0,108); ax.legend(frameon=False,fontsize=9,loc="lower right")
    ax.grid(axis="y",alpha=0.3); plt.tight_layout(); plt.show()
# your held-out numbers:
plot_coverage([55,38],[(38,71),(23,54)],[95,95],[(83,99),(83,99)])
