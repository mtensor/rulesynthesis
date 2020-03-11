data_mini_scan_train = [['dax', 'RED'], 
                ['lug', 'BLUE'], 
                ['wif', 'GREEN'], 
                ['zup', 'YELLOW'],
                ['dax fep', 'RED RED RED'],
                ['lug fep', 'BLUE BLUE BLUE'],
                ['wif blicket dax', 'GREEN RED GREEN'],
                ['lug blicket wif', 'BLUE GREEN BLUE'],
                ['dax kiki lug', 'BLUE RED'],
                ['lug kiki wif', 'GREEN BLUE'],
                ['lug fep kiki wif', 'GREEN BLUE BLUE BLUE'],
                ['lug kiki wif fep', 'GREEN GREEN GREEN BLUE'],
                ['wif kiki dax blicket lug', 'RED BLUE RED GREEN'],
                ['wif blicket dax kiki lug', 'BLUE GREEN RED GREEN']]

data_mini_scan_test =  [['zup fep', 'YELLOW YELLOW YELLOW'],
                ['zup blicket lug', 'YELLOW BLUE YELLOW'],
                ['zup kiki dax', 'RED YELLOW'],
                ['zup fep kiki lug', 'BLUE YELLOW YELLOW YELLOW'],
                ['wif kiki zup fep', 'YELLOW YELLOW YELLOW GREEN'],
                ['lug kiki wif blicket zup', 'GREEN YELLOW GREEN BLUE'],
                ['zup blicket wif kiki dax fep', 'RED RED RED YELLOW GREEN YELLOW'],
                ['zup blicket zup kiki zup fep', 'YELLOW YELLOW YELLOW YELLOW YELLOW YELLOW']]


from agent import Example


examples_train = {Example(inp.split(' '), out.split(' ') ) for inp, out in data_mini_scan_train}

examples_test = {Example(inp.split(' '), out.split(' ') ) for inp, out in data_mini_scan_test}