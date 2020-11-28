from numpy import arange
from src.linguistic_var import LinguisticVar
from src.membership_functions import create_bell, create_s, create_gamma, create_triangular
from src.inference_system import FuzzySystem, LARSEN
from src.defuzzification import coa, mom, som, boa
from src.utils import plot_fuzzy_set


PARTICIPANTES = ["Nico", "Josy", "Terre", "Flo"]
PUNTUACIONES = [[8, 9, 7], [7, 7, 8], [6, 6, 5], [6, 4, 5]]

if __name__ == '__main__':
    prod_quality = LinguisticVar("Producto"
                                 , [ "buena", "excelente" ]
                                 , [ create_bell(5, 2), create_s(6, 10) ]
                                 , universe = arange(0, 10.1, 0.1))
    
    cook_quality = LinguisticVar("Elaboración"
                                 , [ "mala", "buena", "excelente" ]
                                 , [ lambda value : 1 - create_gamma(0, 6)(value), create_triangular(4, 6, 8), create_s(6, 10) ]
                                 , universe = arange(0, 10.1, 0.1))
    
    plating_quality = LinguisticVar("Emplatado"
                                    , [ "malo", "bueno", "excelente"] 
                                    , [ lambda value : 1 - create_gamma(0, 6)(value), create_triangular(4, 6, 8), create_s(6, 10) ]
                                    , universe = arange(0, 10.1, 0.1))
                                    

    valoration = LinguisticVar("Valoración"
                               , ["mala", "buena", "excelente"]
                               , [ lambda value : 1 - create_gamma(0, 6)(value), create_triangular(4, 6, 8), create_s(6, 10) ]
                               , universe = arange(0, 10.1, 0.1))
    
    masterchef_judge = FuzzySystem([ prod_quality, cook_quality, plating_quality ], [ valoration ])
    
    masterchef_judge += (cook_quality, plating_quality), (valoration, ), \
                        (cook_quality.excelente & plating_quality.excelente) >> valoration.excelente

    masterchef_judge += (cook_quality, plating_quality), (valoration, ), \
                        (cook_quality.excelente & plating_quality.bueno) >> valoration.buena
    
    masterchef_judge += (cook_quality, ), (valoration, ), \
                        cook_quality.mala >> valoration.mala
    
    masterchef_judge += (prod_quality, cook_quality, plating_quality), (valoration, ), \
                        (prod_quality.excelente & cook_quality.buena & plating_quality.excelente) >> valoration.excelente
    
    masterchef_judge += (prod_quality, cook_quality, plating_quality), (valoration, ), \
                        (prod_quality.buena & cook_quality.buena & plating_quality.malo) >> valoration.mala

    masterchef_judge += (prod_quality, cook_quality, plating_quality), (valoration, ), \
                        (prod_quality.buena & cook_quality.excelente & plating_quality.malo) >> valoration.buena
    
    masterchef_judge += (cook_quality, plating_quality), (valoration, ), \
                        (cook_quality.buena & plating_quality.bueno) >> valoration.buena

    masterchef_judge += (cook_quality, plating_quality), (valoration, ), \
                        (cook_quality.buena & plating_quality.excelente) >> valoration.buena

    maxi = 0
    winner = 0
    for i, scores in enumerate(PUNTUACIONES):
        val = mom(masterchef_judge.inference(*scores)[valoration])
        if val > maxi:
            winner = i
            maxi = val
    print(f"El ganador de la prueba es {PARTICIPANTES[winner]}")