from neural_network import Network
from turtle import Turtle, mainloop
from math import sin, pi

# FUNCTION INFO:
FUNC = sin
DOMAIN: tuple = (0, 2 * pi)
STEP: float = pi / 12
V_RANGE: tuple = (-1, 1)

# NETWORK CONFIG:
HIDDEN_LAYERS: int = 2
NEURONS_PER_LAYER: int = 8


def main():
    ai = Network(HIDDEN_LAYERS, NEURONS_PER_LAYER, 1, 1)
    turtle: Turtle
    x: float
    y: float
    loss: float
    min_loss: float = 1e6

    try:
        while min_loss > 0.005:
            for layer_index in range(HIDDEN_LAYERS + 1):
                for q in range(1, 51):
                    ai.randomize(q * 0.001, layer_index)
                    loss = 0
                    x = DOMAIN[0]
                    
                    while x < DOMAIN[1]:
                        y = ai.test_compute([x, ], output_activation=lambda output: output)[0]
                        loss += abs(y - FUNC(x))
                        x += STEP
                    
                    # MEDIUM VALUE OF THE LOSS
                    loss /= (DOMAIN[1] - DOMAIN[0]) // STEP
                    # PERCENT OF THE FUCTION'S RANGE
                    loss /= (V_RANGE[1] - V_RANGE[0])

                    if loss < min_loss:
                        ai.apply()
                        min_loss = loss
                        print(f'{layer_index}-{q}: {round(min_loss * 100, 1)} %')
    except:
        pass
        

    turtle = Turtle('turtle')
    turtle.speed(2)

    # DRAW FUNCTION
    turtle.color('green')
    turtle.up()
    turtle.goto(-200, 0)
    turtle.down()

    x = 0
    while x < 2 * pi:
        y = FUNC(x)

        turtle.goto(-200 + x * 50, y * 50)
        x += pi / 12
    
    # DRAW BEST FOUND
    turtle.color('orange')
    turtle.up()
    turtle.goto(-200, 0)
    turtle.down()

    x = 0
    while x < 2 * pi:
        y = ai.compute([x, ], output_activation=lambda output: output)[0]

        turtle.goto(-200 + x * 50, y * 50)
        x += pi / 12
    
    mainloop()


if __name__ == '__main__':
    main()
