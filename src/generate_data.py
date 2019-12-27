from data_generation.dimacs_generators import DimacsGenerator

generator = DimacsGenerator("../out", percentage_sat=0.50)

generator.generate(1000)



