# radiotools

<p align="center">
    <img src="./assets/BINGO_LOGO-cor.png" width=20%
        alt = "Release Status">
</p>

Radiotools é um pacote em python cujo objetivo é prover o usuário de ferramentas para determinar algumas propriedades fundamentais de uma observação radioastronômica. Uma vez definido um objeto da classe `instrument` o usuário pode acessar o planejamento de observações em trânsito, com visualização em mapas em diferentes coordenadas celestes e diferentes projeções, de maneira direta.

O pacote ainda permite correlacionar as informações das observações com dados observacionais armazenados localmente em diferentes formatos, assim que o usuário definir a classe de `backend` para o seu instrumento.


## Requisitos

- `numpy`: The fundamental package for scientific computing with Python
- `scipy`: SciPy (pronounced “Sigh Pie”) is a Python-based ecosystem of open-source software for mathematics, science, and engineering. In particular
- `pandas`: pandas aims to be the fundamental high-level building block for doing practical, real world data analysis in Python. Additionally, it has the broader goal of becoming the most powerful and flexible open source data analysis / manipulation tool available in any language.
- `astropy`: The Astropy Project is a community effort to develop a core package for astronomy using the Python programming language and improve usability, interoperability, and collaboration between astronomy Python packages. The core astropy package contains functionality aimed at professional astronomers and astrophysicists, but may be useful to anyone developing astronomy software. The Astropy Project also includes "affiliated packages," Python packages that are not necessarily developed by the core development team, but share the goals of Astropy, and often build from the core package's code and infrastructure.
- `astroquery`: a set of tools for querying astronomical web forms and databases.
- `pytz`: [pytz](http://http://pytz.sourceforge.net/#introduction) brings the Olson tz database into Python. This library allows accurate and cross platform timezone calculations using Python 2.4 or higher. It also solves the issue of ambiguous times at the end of daylight saving time, which you can read more about in the Python Library Reference (datetime.tzinfo).
- `skyfield`: Skyfield computes positions for the stars, planets, and satellites in orbit around the Earth. Its results should agree with the positions generated by the United States Naval Observatory and their Astronomical Almanac to within 0.0005 arcseconds (half a “mas” or milliarcsecond).
