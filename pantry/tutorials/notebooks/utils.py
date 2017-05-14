def get_cost():
    costs = []
    with open('./resultor/costs.txt') as costf:
        costs = [float(cost.rstrip()) for cost in costf]
    return costs
