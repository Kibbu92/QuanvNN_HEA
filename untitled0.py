# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 16:18:07 2023

@author: andca
"""

import jax

if __name__ == '__main__':
    
    
    rng_key = jax.random.PRNGKey(42)

    for i in range(5):
        
        print('')
    
    
        for h in range(5):
            a = jax.random.uniform(rng_key+h)
            print(a)