from neural_network import Network
from turtle import Turtle, mainloop
from math import sin, pi

# MATH FUNCTION INFO:
FUNC = sin
DOMAIN: tuple = (0, 2 * pi)
SET_OF_VALUES: tuple = (-1, 1)
STEP: float = pi / 24

# NETWORK CONFIG:
HIDDEN_LAYERS: int = 2
NEURONS_PER_LAYER: int = 8
TARGET_LOSS: float = 0.025

def main():
    ai = Network(HIDDEN_LAYERS, NEURONS_PER_LAYER, 1, 1)
    turtle: Turtle
    x: float
    y: float
    loss: float
    min_loss: float = 1e6

    try:
        while min_loss > TARGET_LOSS:
            for layer_index in range(HIDDEN_LAYERS + 1):
                # for q in range(1, 11):
                # ai.randomize(q * 0.001, layer_index)
                ai.randomize(0.004, layer_index)
                loss = 0
                x = DOMAIN[0]
                
                while x < DOMAIN[1]:
                    y = ai.test_compute([x, ], output_activation=lambda output: output)[0]
                    loss += abs(y - FUNC(x))
                    x += STEP
                
                # Medium value of the loss per one step
                loss /= (DOMAIN[1] - DOMAIN[0]) // STEP
                # Converting to a percent of the function's set of values
                loss /= (SET_OF_VALUES[1] - SET_OF_VALUES[0])

                # If better adjustment found:
                if loss < min_loss:
                    ai.apply()
                    min_loss = loss
                    # print(f'[{layer_index}-{q}] Loss: {round(min_loss * 100, 1)} %')
                    print(f'[{layer_index}] Loss: {round(min_loss * 100, 1)} %')
    
    except:
        pass
        

    turtle = Turtle('turtle')
    turtle.speed(0)

    # DRAW FUNCTION
    turtle.color('green')
    turtle.up()
    turtle.goto(-200, 0)
    turtle.down()

    x = DOMAIN[0]
    while x < DOMAIN[1]:
        y = FUNC(x)

        turtle.goto(-200 + x * 50, y * 50)
        x += pi / 12
    
    # DRAW BEST FOUND
    turtle.color('orange')
    turtle.up()
    turtle.goto(-200, 0)
    turtle.down()

    x = DOMAIN[0]
    while x < DOMAIN[1]:
        y = ai.compute([x, ], output_activation=lambda output: output)[0]

        turtle.goto(-200 + x * 50, y * 50)
        x += STEP
    
    mainloop()


if __name__ == '__main__':
    main()
