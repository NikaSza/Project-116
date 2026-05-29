#example of condition to be changed later/ add similar functions below
def momentum_mask(pt_min=1000): # change the treshold if you want more or less stict cuts
    pt = tree["B0_PT"].array(library="np")
    return np.isfinite(pt) & (pt > pt_min)

def mcorr_mask(mcr_min=1554, mcr_max=5743):
    mcr = tree["B0_MCORR"].array(library="np")
    return np.isfinite(mcr), np.isfinite(mcr_max) & (mcr_min<mcr<mcr_max)

def b0_ipchi2_ownpv(ipchi_max=5): # change the treshold if you want more or less stict cuts
    ipchi2 = tree["B0_IPCHI2_OWNPV"].array(library="np")
    return np.isfinite(ipchi2) & (ipchi2 < ipchi_max)

# IS MUON/HAS MUON:

def kplus_ismuon(k_ismu=1): # change the treshold if you want more or less stict cuts
    k_mu = tree["Kplus_isMuon"].array(library="np")
    return np.isfinite(k_mu) & (k_mu < k_ismu)

def kplus_hasmuon(k_hasmuon=1): # change the treshold if you want more or less stict cuts
    k_mu1 = tree["Kplus_hasMuon"].array(library="np")
    return np.isfinite(k_mu1) & (k_mu1 > k_hasmuon)

def piminus_ismuon(pi_ismu=1): # change the treshold if you want more or less stict cuts
    pi_mu = tree["piminus_isMuon"].array(library="np")
    return np.isfinite(pi_mu) & (pi_mu < pi_ismu)

def piminus_hasmuon(pi_hasmuon=1): # change the treshold if you want more or less stict cuts
    pi_mu1 = tree["piminus_hasMuon"].array(library="np")
    return np.isfinite(pi_mu1) & (pi_mu1 > pi_hasmuon)
                                
# PROBABILITY THAT STUFF IS KAONS:

def kaon_probability_mask(k_threshold=0.85): # change the treshold if you want more or less stict cuts
    kplus = tree["Kplus_ProbNNk"].array(library="np")
    mask = np.isfinite(kplus) & (kplus > k_threshold)
    return mask

def piminus_prob_k(pi_k_max=0.15): # change the treshold if you want more or less stict cuts
    pi_k = tree["piminus_ProbNNk"].array(library="np")
    return np.isfinite(pi_k) & (pi_k < pi_k_max)

def muminus_prob_k

def muplus_prob_k

# PROBABILITY THAT STUFF IS MUONS:

def kplus_probmu(kp_mu_max=0.15): # change the treshold if you want more or less stict cuts
    kp_mu = tree["Kplus_ProbNNmu"].array(library="np")
    return np.isfinite(kp_mu) & (kp_mu < kp_mu_max)

def piminus_prob_mu

def muminus_prob_mu

def muplus_prob_mu

# PROBABILITY THAT STUFF IS ELECTRONS:

def kp_prob_e(kp_e_max=0.15): # change the treshold if you want more or less stict cuts
    kp_e = tree["Kplus_ProbNNe"].array(library="np")
    return np.isfinite(kp_e) & (kp_e < kp_e_max)

def muplus_prob_e(muplus_e_max=0.15): # change the treshold if you want more or less stict cuts
    muplus_e = tree["muplus_ProbNNe"].array(library="np")
    return np.isfinite(muplus_e) & (muplus_e < muplus_e_max)

def muminus_prob_e(muminus_e_max=0.15): # change the treshold if you want more or less stict cuts
    muminus_e = tree["muminus_ProbNNe"].array(library="np")
    return np.isfinite(muminus_e) & (muminus_e < muminus_e_max)

def piminus_prob_e

# PROBABILITY THAT STUFF IS PROTONS:

def kp_prob_p(kp_p_max=0.15): # change the treshold if you want more or less stict cuts
    kp_p = tree["Kplus_ProbNNp"].array(library="np")
    return np.isfinite(kp_p) & (kp_p < kp_p_max)

def muminus_prob_p(muminus_p_max=0.15): # change the treshold if you want more or less stict cuts
    muminus_p = tree["muminus_ProbNNp"].array(library="np")
    return np.isfinite(muminus_p) & (muminus_p < muminus_p_max)

def muplus_prob_p(muplus_p_max=0.15): # change the treshold if you want more or less stict cuts
    muplus_p = tree["muplus_ProbNNp"].array(library="np")
    return np.isfinite(muplus_p) & (muplus_p < muplus_p_max)

def piminus_prob_p

# PROBABILITY THAT STUFF IS PIONS:

def kp_prob_pi(kp_pi_max=0.15): # change the treshold if you want more or less stict cuts
    kp_pi = tree["Kplus_ProbNNpi"].array(library="np")
    return np.isfinite(kp_pi) & (kp_pi < kp_pi_max)

def muminus_prob_pi(muminus_pi_max=0.15): # change the treshold if you want more or less stict cuts
    muminus_pi = tree["muminus_ProbNNpi"].array(library="np")
    return np.isfinite(muminus_pi) & (muminus_pi < muminus_pi_max)

def muplus_prob_pi(muplus_pi_max=0.15): # change the treshold if you want more or less stict cuts
    muplus_pi = tree["muplus_ProbNNpi"].array(library="np")
    return np.isfinite(muplus_pi) & (muplus_pi < muplus_pi_max)

def piminus_prob_pi



#combine conditions -add new masks to variable list here
def combine_masks(k_threshold=0.85, pt_min=1000, mcr_min=1554, mcr_max=5743, ipchi_max=5, kp_e_max=0.15, kp_p_max=0.15, kp_pi_max=0.15, kp_mu_max=0.15, muplus_e_max=0.15, muminus_e_max=0.15, muplus_p_max=0.15, muminus_pmax=0.15, muplus_pi_max=0.15, muminus_pi_max=0.15, pi_ismu=1, pi_hasmuon=1, k_ismu=1, k_hasmuon=1, pi_k_max=0.15):
    mask = (
        kaon_probability_mask(k_threshold) &
        momentum_mask(pt_min) 
        & 
    )
    return mask
