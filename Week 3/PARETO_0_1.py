# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 01:26:45 2025

@author: MARCELOFGB
"""

from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter

pf = get_problem("truss2d").pareto_front()

sc = Scatter(title="Pareto-front")
sc.add(pf, s=80, facecolors='none', edgecolors='r')
sc.add(pf, plot_type="line", color="black", linewidth=2)
sc.show()