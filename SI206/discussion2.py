# Your name: Nicolas Newberry
# Your student id: 41052667
# Your email: nnicolas@umich.edu
# List who or what you worked with on this homework: ChatGPT
# If you used Generative AI, say that you used it and also how you used it.
# I used ChatGPT for help with drawing the mouth as I wasn't entirely sure how to create the semicircle and angle it correctly.

from turtle import *

def draw_circle(turtle, x_position, y_position, fill_color, radius):
    turtle.penup()
    turtle.goto(x_position, y_position - radius)
    turtle.pendown()
    turtle.begin_fill()
    turtle.fillcolor(fill_color)
    turtle.circle(radius)
    turtle.end_fill()

def draw_rectangle(turtle, x_position, y_position, fill_color, width, height):
    turtle.penup()
    turtle.goto(x_position, y_position)
    turtle.setheading(0)
    turtle.pendown()
    turtle.begin_fill()
    turtle.fillcolor(fill_color)
    for _ in range(2):
        turtle.forward(width)
        turtle.left(90)
        turtle.forward(height)
        turtle.left(90)
    turtle.end_fill()

def draw_emoji(turtle):
    #Draw face
    draw_circle(turtle, 0, 0, "yellow", 100)

    #Draw left eye
    draw_circle(turtle, -40, 50, "black", 15)

    #Draw right eye
    draw_circle(turtle, 40, 50, "black", 15)

    #Draw eyebrows
    draw_rectangle(turtle, -55, 70, "brown", 30, 5)  # Left eyebrow
    draw_rectangle(turtle, 25, 70, "brown", 30, 5)   # Right eyebrow

    #Draw the mouth
    turtle.penup()
    turtle.goto(-50, -30)
    turtle.setheading(-60)
    turtle.pendown()
    turtle.width(5)
    turtle.circle(60, 120)

def main():
    screen = Screen()
    screen.bgcolor("white")

    #Create Turtle object
    my_turtle = Turtle()

    #Call draw_emoji
    draw_emoji(my_turtle)

    #Exit on click
    screen.exitonclick()

if __name__ == '__main__':
    main()
