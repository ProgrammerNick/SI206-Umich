# Your name: Nicolas Newberry
# Your student id: 41052667
# Your email: nnicolas@umich.edu
# List who or what you worked with on this homework: ChatGPT
# If you used Generative AI, say that you used it and also how you used it.

"'I used Generative AI for some small pieces of the syntax. I wasn't fully familiar with randomization syntax. I know that this is something the professor wanted us to practice with, so I used GenAI in the development of the randomization aspect of the snowflakes and in importing the library. I also wanted to make my snowflakes unique (rather than just drawing circles), so I used GenAI to help me with creating an object that would be interesting to see. Lastly, I was having difficulties with aligning circles for the mouth of the snowman, so I used GenAI to give me a better solution on creating the mouth as a half-circle, which worked out nicely'"

from turtle import *
import random

def draw_circle(turtle, x_position, y_position, radius, fill_color):
    "'Draws the circles for the snowman'"
    turtle.penup()
    turtle.goto(x_position, y_position - radius)
    turtle.pendown()
    turtle.fillcolor(fill_color)
    turtle.begin_fill()
    turtle.circle(radius)
    turtle.end_fill()

    

def draw_rectangle(turtle, x_position, y_position, width, height, fill_color, angle=0, no_border=False):
    "'Draws the rectangle for the scarf'"
    turtle.penup()
    turtle.goto(x_position, y_position)
    turtle.setheading(angle)
    turtle.pendown()
    turtle.fillcolor(fill_color)

    # set the border color of the turtle equal to the fill color so you can't recognize the rectangle shape for the scarf
    if no_border:
        turtle.pencolor(fill_color)

    turtle.begin_fill()
    for _ in range(2):
        turtle.forward(width)
        turtle.left(90)
        turtle.forward(height)
        turtle.left(90)
    turtle.end_fill()
    turtle.setheading(0)

    # set the border color back to black
    if no_border:
        turtle.pencolor("black")

def draw_triangle(turtle, x_position, y_position, side_length, fill_color):
    "'Draws the triangle for the nose'"
    turtle.penup()
    turtle.goto(x_position, y_position)
    turtle.pendown()
    turtle.fillcolor(fill_color)
    turtle.begin_fill()
    for _ in range(3):
        turtle.forward(side_length)
        turtle.left(120)
    turtle.end_fill()

def draw_snowflake(turtle, x_position, y_position):
    "'Draws the snowflakes'"
    turtle.penup()
    turtle.goto(x_position, y_position)
    turtle.pendown()
    turtle.fillcolor("green")
    turtle.begin_fill()
    turtle.circle(2)
    turtle.goto(x_position -1, y_position + 1)

    for _ in range(6):
        for _ in range(3):
            turtle.forward(20)
            turtle.backward(20)
            turtle.right(45)
        turtle.left(135)
        turtle.right(60)
    
    turtle.end_fill()

def draw_first_n(turtle, x_position, y_position):
    """Draws the first 'N'."""
    turtle.penup()
    turtle.goto(x_position, y_position)  # Starting position
    turtle.pendown()
    turtle.pencolor("blue")
    turtle.pensize(20)

    #Vertical bar
    turtle.setheading(90)
    turtle.forward(100)
    #Diagonal bar
    turtle.setheading(310)
    turtle.forward(120)
    #Vertical bar
    turtle.setheading(90)
    turtle.forward(100)
    turtle.penup()

def draw_second_n(turtle, x_position, y_position):
    """Draws the second 'N'."""
    turtle.penup()
    turtle.goto(x_position, y_position)  # Starting position
    turtle.pendown()
    turtle.pencolor("blue")
    turtle.pensize(20)

    #Vertical bar
    turtle.setheading(90)
    turtle.forward(100)
    #Diagonal bar
    turtle.setheading(310)
    turtle.forward(120)
    #Vertical bar
    turtle.setheading(90)
    turtle.forward(100)
    turtle.penup()

def draw_bird(turtle, x_position, y_position):
    # Move the turtle to the starting position
    turtle.penup()
    turtle.goto(x_position, y_position)
    turtle.pendown()

    # Set the width of the pen for better visibility
    turtle.pensize(2)

    # Draw the body
    turtle.circle(20)  # Small circle for the body

    # Draw the head
    turtle.penup()
    turtle.goto(x_position + 25, y_position + 20)
    turtle.pendown()
    turtle.circle(10)  # Smaller circle for the head

    # Draw the beak
    turtle.penup()
    turtle.goto(x_position + 30, y_position + 25)
    turtle.setheading(30)
    turtle.pendown()
    turtle.goto(x_position + 35, y_position + 30)

    # Draw the eye
    turtle.penup()
    turtle.goto(x_position + 30, y_position + 30)
    turtle.pendown()
    turtle.circle(2)

    turtle.penup()
    turtle.goto(x_position, y_position)
    turtle.goto(x_position - 5, y_position + 5)
    turtle.setheading(0)
    turtle.pendown()
    turtle.goto(x_position - 14, y_position)  # Arc for the wing
    turtle.goto(x_position + 7, y_position - 7)
    turtle.goto(x_position, y_position)

    turtle.penup()
    turtle.goto(x_position, y_position + 20)
    
    turtle.pendown()
    turtle.goto(x_position + 5, y_position + 10)
    turtle.goto(x_position + 10, y_position + 20)



def draw_winter_scene(turtle, screen_width, screen_height):
    "'Draw the snowman'"
    #draw hills
    draw_circle(turtle, -400, -500, 500, "red")
    draw_circle(turtle, 300, -500, 500, "green")
    #draw body
    draw_circle(turtle, 0, -150, 120, "white")
    draw_circle(turtle, 0, 0, 80, "white")
    draw_circle(turtle, 0, 80, 60, "white")
    #draw hat
    draw_rectangle(turtle, -35, 120, 80, 20, "black")
    draw_rectangle(turtle, -20, 130, 40, 60, "black")
    #draw eyes
    draw_circle(turtle, -10, 100, 5, "black")
    draw_circle(turtle, 10, 100, 5, "black")
    #draw nose
    draw_triangle(turtle, -10, 70, 20, "orange")
    #draw scarf
    draw_rectangle(turtle, 20, 20, 30, 70, "red", 330, no_border=True)
    draw_rectangle(turtle, 45, 35, 85, 30, "red", 180, no_border=True)
    draw_rectangle(turtle, -40, 5, 30, 70, "red", 30, no_border=True)
    draw_rectangle(turtle, 130, 65, 30, 70, "red", 100, no_border=True)
    draw_rectangle(turtle, 135, 20, 30, 70, "red", 40, no_border=True)
    draw_rectangle(turtle, 110, 0, 85, 30, "red", 150, no_border=True)
    draw_rectangle(turtle, 158, -30, 30, 70, "red", 30, no_border=True)

    #draw arms
    turtle.penup()
    turtle.goto(200, 200)
    turtle.pendown()
    draw_rectangle(turtle, 40, -30, 100, 5, "brown")
    draw_rectangle(turtle, 110, -40, 30, 5, "brown", 90)
    draw_rectangle(turtle, -140, -30, 100, 5, "brown")
    draw_rectangle(turtle, -110, -40, 30, 5, "brown", 90)

    #draw smile
    turtle.penup()
    turtle.goto(-5, 50)
    turtle.setheading(-60)
    turtle.pendown()
    turtle.circle(10, 120)


    # draw random snowflakes
    for _ in range(20):
        x_position = random.randrange(-screen_width, screen_width)
        y_position = random.randrange(-screen_height, screen_height)

        draw_snowflake(turtle, x_position, y_position)
    
    for _ in range(5):
        x_position = random.randrange(-screen_width + 20, screen_width - 20)
        y_position = random.randrange(100, screen_height - 20)

        turtle.setheading(0)
        draw_bird(turtle, x_position, y_position)



def main():
    
    screen = Screen()

    screen_width = screen.window_width() // 2  # Half width for positive/negative range
    screen_height = screen.window_height() // 2  # Half height for positive/negative range

    screen.bgcolor("sky blue")
    nicolas = Turtle()
    nicolas.speed("fastest")
    draw_winter_scene(nicolas, screen_width, screen_height) #draws the winter scene and snowman
    draw_first_n(nicolas, 150, -300) #draws the first N
    draw_second_n(nicolas, 270, -300) #draws the second N
    screen.exitonclick()



if __name__ == '__main__':
    main()