# standax
Type Ia Supernovae Standardisation with jax

To install the package, just do: 

    git clone https://github.com/mginolin/standax.git
    cd standax
    pip install .

This is a simplified version of the code that was used to fit for standardisation in the ZTF DR2 analysis (Rigault+25a, Ginolin+25a,b). The full code uses optimisation methods that will be realeased with Khun+ (in prep), which do not change the outcome of the code but make it faster. The concept of 'total $\chi^2$' minimisation on which `standax` is based, is explained in Section 4.2 of Ginolin+25b and Khun+ (in prep). If you use the code please cite those two papers.

There are a few examples notebooks in `/notebooks/` :
- `total_chi2.ipynb` and `totalchi2_and_standardisation.ipynb` showcase the principal of the total $\chi^2$ fit for line-fitting and applied to a simple simulated SN standardisation problem.
- `Standax_standardisation.ipynb` shows a use case of `standax` on the ZTF DR2 data, whith additional advanced features.
- `Standardisation_Ginolin25b.ipynb` shows how the standardisation parameters of Ginolin+25b were obtained, with the `standax.hubble` module, which makes use of the `standax.standardisation` module but packages it for easier use.
- `FigA2_Ginolin25b.ipynb` reproduces the Fig. A2 of Ginolin+25b, which compares a simple loglkelihood minimisation to the total $\chi^2$ one on `skysurvey` SN simulations.
- `Appendix_Ginolin25a.ipynb` reproduces the figures of Appendix A of Ginolin+25a.
