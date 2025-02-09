from generate import generate_recipe

if __name__ == "__main__":
    user_input = input("Enter ingredients (comma-separated): ")
    recipe = generate_recipe(user_input)
    print("\nGenerated Recipe:\n", recipe)