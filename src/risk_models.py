import math


def ped_cyclists_injury_probability(a, b, c, vel, age):  # Eq. 5
    prob = 1/(1+math.exp(-1*(a + b*vel + c*age)))
    return prob

def car_passenger_injury_probability(a, b, vel):  # Eq. 6
    prob = 1/(1+math.exp(-1*(a + b*vel)))
    return prob
