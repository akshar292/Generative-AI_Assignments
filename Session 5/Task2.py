def restaurant_chatbot(restaurant_name, cuisine):
    return f"Yes, {restaurant_name} serves {cuisine} cuisine."

restaurant = input("Enter restaurant name: ")
cuisine = input("Enter cuisine: ")

response = restaurant_chatbot(restaurant, cuisine)
print(response)