import os

string2 = ''
for j in range(10):
    string = ''
    for i in range(10):
        string += f'outputs/hw2_2/{i}_{j:03d}.png '
    os.system(f'convert -append {string} outputs/{j}_out.png')
    string2 += f'outputs/{j}_out.png '
os.system(f'convert +append {string2} out.png')
