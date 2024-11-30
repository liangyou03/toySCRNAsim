import numpy as np
import pandas as pd

def simple_sim(N=200, G=1500, seed=None):
    """
    Generate simplified scRNA-seq data with negative binomial distribution

    Parameters:
    - N (int): number of cells
    - G (int): number of genes
    - seed (int, optional): random seed

    Return:
    - pandas.DataFrame: Simulated scRNA-seq data
    """
    if seed is not None:
        np.random.seed(seed)

    # Simulate gene means from lognormal distribution
    gene_means = np.random.lognormal(mean=1, sigma=0.5, size=G)

    # Simulate gene dispersion from uniform distribution
    gene_dispersion = np.random.uniform(0.1, 1.0, size=G)

    # Simulate count data using negative binomial distribution  
    counts = np.zeros((G, N), dtype=int)

    for gene in range(G):
        mu = gene_means[gene]
        dispersion = gene_dispersion[gene]
        size = 1.0 / dispersion
        p = size / (size + mu)
        counts[gene, :] = np.random.negative_binomial(n=size, p=p, size=N)

    gene_names = [f"Gene_{i+1}" for i in range(G)]
    cell_names = [f"Cell_{i+1}" for i in range(N)]

    df = pd.DataFrame(data=counts, index=gene_names, columns=cell_names)

    return df

def splat_sim(N=200, G=1500, alpha0=1, beta0=1, p_outlier=0.05, xi=1, omega=1,
                   mu_L=30, sigma_L=0.2, phi0=0.1, d=7, b=1):
    """
    Simulate scRNA-seq count data using the Splat simulation model.

    Parameters:
    N : int
        Number of cells.
    G : int
        Number of genes.
    alpha0 : float, optional
        Shape parameter for the gamma distribution of gene means.
    beta0 : float, optional
        Rate parameter for the gamma distribution of gene means.
    p_outlier : float, optional
        Probability that a gene is a high expression outlier.
    xi : float, optional
        Location parameter for the log-normal distribution of inflation factors.
    omega : float, optional
        Scale parameter for the log-normal distribution of inflation factors.
    mu_L : float, optional
        Mean (in log-space) for the log-normal distribution of library sizes.
    sigma_L : float, optional
        Standard deviation (in log-space) for the log-normal distribution of library sizes.
    phi0 : float, optional
        Dispersion parameter controlling the overdispersion in counts.
    d : float, optional
        Midpoint parameter for the dropout logistic function.
    b : float, optional
        Shape parameter for the dropout logistic function.

    Returns:
    pd.DataFrame
        A DataFrame containing the simulated count matrix with genes as rows and cells as columns.
    """
    # Step 1: Simulate gene means μ_g ~ Gamma(α0, β0)
    mu_g = np.random.gamma(shape=alpha0, scale=1 / beta0, size=G)

    # Step 2: Introduce high expression outliers
    outlier_genes = np.random.rand(G) < p_outlier
    median_mu_g = np.median(mu_g)
    num_outliers = np.sum(outlier_genes)
    if num_outliers > 0:
        inflation_factors = np.random.lognormal(mean=xi, sigma=omega, size=num_outliers)
        mu_g[outlier_genes] = median_mu_g * inflation_factors

    # Step 3: Simulate library sizes L_c ~ LogNormal(μ_L, σ_L)
    L_c = np.random.lognormal(mean=mu_L, sigma=sigma_L, size=N)
    L_c_norm = L_c / np.mean(L_c)
    mu_gc = np.outer(mu_g, L_c_norm)  # Gene x Cell matrix

    # Step 4: Simulate overdispersion
    phi0 = max(phi0, 1e-8)  # Avoid division by zero
    shape_param = 1 / phi0
    scale_param = mu_gc * phi0
    lambda_gc = np.random.gamma(shape=shape_param, scale=scale_param)

    # Step 6: Generate counts from Poisson(lambda_gc)
    counts_gc = np.random.poisson(lam=lambda_gc)

    # Step 7: Model zero-inflation (dropouts)
    mu_gc_nonzero = mu_gc + 1e-8  # Avoid log of zero
    log2_mu_gc = np.log2(mu_gc_nonzero)
    P_zero_gc = 1 / (1 + np.exp(-b * (log2_mu_gc - d)))
    dropout_gc = np.random.rand(G, N) < P_zero_gc
    counts_gc[dropout_gc] = 0

    # Create DataFrame
    cells = ['Cell{}'.format(i + 1) for i in range(N)]
    genes = ['Gene{}'.format(i + 1) for i in range(G)]
    df_counts = pd.DataFrame(counts_gc, index=genes, columns=cells)

    return df_counts


def sim_with_cell_types(N=200, G=1500, K=4, alpha0=2, beta0=0.5, p_outlier=0.05, xi=1, omega=1,
                        mu_L=11, sigma_L=0.2, phi0=0.1, d=10, b=1,
                        de_prob=0.3, de_factor=(5, 10)):
    """
    Simulate scRNA-seq count data with cell types and gene DE labels.

    Parameters:
    N : int
        Total number of cells.
    G : int
        Number of genes.
    K : int, optional
        Number of cell types.
    alpha0 : float, optional
        Shape parameter for the gamma distribution of gene means.
    beta0 : float, optional
        Rate parameter for the gamma distribution of gene means.
    p_outlier : float, optional
        Probability that a gene is a high expression outlier.
    xi : float, optional
        Location parameter for the log-normal distribution of inflation factors.
    omega : float, optional
        Scale parameter for the log-normal distribution of inflation factors.
    mu_L : float, optional
        Mean (in log-space) for the log-normal distribution of library sizes.
    sigma_L : float, optional
        Standard deviation (in log-space) for the log-normal distribution of library sizes.
    phi0 : float, optional
        Dispersion parameter controlling the overdispersion in counts.
    d : float, optional
        Midpoint parameter for the dropout logistic function.
    b : float, optional
        Shape parameter for the dropout logistic function.
    de_prob : float, optional
        Probability that a gene is differentially expressed between cell types.
    de_factor : tuple, optional
        Range of fold changes for differentially expressed genes.

    Returns:
    df_counts : pd.DataFrame
        A DataFrame containing the simulated count matrix with genes as rows and cells as columns.
    cell_labels : pd.Series
        A Series containing the cell type labels for each cell.
    gene_de_info : pd.DataFrame
        A DataFrame indicating which genes are differentially expressed in which cell types.
    """
    # Step 1: Simulate gene means μ_g ~ Gamma(α0, β0)
    mu_g = np.random.gamma(shape=alpha0, scale=1 / beta0, size=G)

    # Step 2: Introduce high expression outliers
    outlier_genes = np.random.rand(G) < p_outlier
    median_mu_g = np.median(mu_g)
    num_outliers = np.sum(outlier_genes)
    if num_outliers > 0:
        inflation_factors = np.random.lognormal(mean=xi, sigma=omega, size=num_outliers)
        mu_g[outlier_genes] = median_mu_g * inflation_factors

    # Step 3: Simulate library sizes L_c ~ LogNormal(μ_L, σ_L)
    L_c = np.random.lognormal(mean=mu_L, sigma=sigma_L, size=N)
    L_c_norm = L_c / np.mean(L_c)

    # Step 4: Simulate cell type assignments
    cell_types = np.repeat(np.arange(K), N // K)
    if len(cell_types) < N:
        cell_types = np.hstack((cell_types, np.random.choice(np.arange(K), N - len(cell_types))))
    np.random.shuffle(cell_types)  # Shuffle cell types

    # Step 5: Simulate differential expression between cell types
    gene_de_info = pd.DataFrame(False, index=['Gene{}'.format(i + 1) for i in range(G)],
                                columns=['CellType{}'.format(k) for k in range(K)])
    mu_g_k = np.zeros((G, K))  # Mean expression of genes in each cell type

    for g in range(G):
        mu_g_base = mu_g[g]
        de_cell_types = np.random.rand(K) < de_prob  # DE in each cell type
        fold_changes = np.ones(K)
        if np.any(de_cell_types):
            # Random fold changes within the specified range for DE cell types
            fc = np.random.uniform(de_factor[0], de_factor[1], size=K)
            up_down = np.random.choice([-1, 1], size=K)
            fc = fc ** up_down
            fc[~de_cell_types] = 1  # No change for non-DE cell types
            fold_changes *= fc
            # Mark DE genes in gene_de_info
            gene_de_info.iloc[g, de_cell_types] = True
        mu_g_k[g, :] = mu_g_base * fold_changes

    # Ensure no negative means
    mu_g_k = np.clip(mu_g_k, a_min=1e-8, a_max=None)

    # Step 6: Adjust for library size and create mu_gc matrix
    mu_gc = np.zeros((G, N))
    for idx_c, k in enumerate(cell_types):
        mu_gc[:, idx_c] = mu_g_k[:, k] * L_c_norm[idx_c]

    # Step 7: Simulate overdispersion
    phi0 = max(phi0, 1e-8)  # Avoid division by zero
    shape_param = 1 / phi0
    scale_param = mu_gc * phi0
    lambda_gc = np.random.gamma(shape=shape_param, scale=scale_param)

    # Step 8: Generate counts from Poisson(lambda_gc)
    counts_gc = np.random.poisson(lam=lambda_gc)

    # Step 9: Model zero-inflation (dropouts)
    mu_gc_nonzero = mu_gc + 1e-8  # Avoid log of zero
    log2_mu_gc = np.log2(mu_gc_nonzero)
    P_zero_gc = 1 / (1 + np.exp(-b * (log2_mu_gc - d)))
    dropout_gc = np.random.rand(G, N) < P_zero_gc
    counts_gc[dropout_gc] = 0

    # Create DataFrame for counts
    cells = ['Cell{}'.format(i + 1) for i in range(N)]
    genes = ['Gene{}'.format(i + 1) for i in range(G)]
    df_counts = pd.DataFrame(counts_gc, index=genes, columns=cells)

    # Create Series for cell labels
    cell_labels = pd.Series(cell_types, index=cells, name='CellType')
    cell_labels = 'CellType' + cell_labels.astype(str)

    # Update gene_de_info index to match genes
    gene_de_info.index = genes

    return df_counts, cell_labels, gene_de_info
