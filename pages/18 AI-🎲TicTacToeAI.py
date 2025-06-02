import streamlit as st
#from predict_page1 import show_predict_page
#from explore_page1 import show_explore_page


#page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))
st.set_page_config(
    page_icon="🧑‍💻"
)


def show_page():

    st.title("TicTacToe AI Games")

    st.write("""### Player Options""")

    px = st.selectbox("Player X",["RANDOM","ABS","MCTS"])
    py = st.selectbox("Player Y",["MCTS","RANDOM","ABS"])
    pg = st.selectbox("Number of Games",range(1,20))

    playermap = {"RANDOM":0,"ABS": 1,"MCTS" :2,"You": 3}

    st.write("""### Board Options""")

    pm = st.selectbox("Row",[3,4,5])
    pn = st.selectbox("Col",[3,4,5])
    pk = st.text_input("How many in a row to win",value="3")

    st.write("""### Computer Options""")

    pr = st.text_input("MCTS (Monte Carlo Tree Search) rollouts",value="500")
    pa = st.text_input("MCTS (Monte Carlo Tree Search) alpha",value="20")

    ok = st.button('Start')


    if ok:
        import numpy as np

        import hw2.mnk_game as mnk
        from hw2.utils import GameStrategy    

        m = pm
        n = pn
        k = pk

        rollouts = int(pr)
        alpha = int(pa)

        num_games = pg
        Xstrat = GameStrategy(playermap[px])
        Ostrat = GameStrategy(playermap[py])
        print_result = True

        state = np.full((m, n), ".")
        player = "X"

        results = {"X wins": 0, "O wins": 0, "draws": 0}
        
        for _ in range(num_games):
            result = mnk.game_loop(
                state, player, k, Xstrat, Ostrat, rollouts, alpha, print_result
            )
            
            if result == 1:
                results["X wins"] += 1
                st.write("X wins")
                st.write(results)
            elif result == -1:
                results["O wins"] += 1
                st.write("O wins")
                st.write(results)

            else:
                results["draws"] += 1
                st.write("Ties")
                st.write(results)
        
        st.write(results)
        again = st.button("Play again?")


        
show_page()

