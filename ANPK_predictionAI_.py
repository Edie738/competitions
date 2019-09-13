{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "NPK predictionAI .ipynb",
      "version": "0.3.2",
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Edie738/competitions/blob/master/ANPK_predictionAI_.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "CODnInYGcafn",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import seaborn as sns\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.model_selection import train_test_split"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tH_V18oKvS02",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from sklearn.tree  import DecisionTreeRegressor"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JXR98iClwJiE",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from sklearn.model_selection import GridSearchCV\n",
        "from sklearn.neighbors import KNeighborsRegressor\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tsn6JqZmhkiN",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "ata = pd.read_csv('https://raw.githubusercontent.com/Edie738/ProjectStuff/master/Soil-Analysis-and-Yield-Prediction-master/Soil-Analysis-and-Yield-Prediction-master/TestingAndTrainingDataSet.csv?token=AM5B2MY2HSDQ53WI5HOLWAS5PO4M4')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ItT5DQSNlEUr",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 347
        },
        "outputId": "ca3bd11f-6495-492c-ad17-7486c1350f70"
      },
      "source": [
        "Data"
      ],
      "execution_count": 157,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>name</th>\n",
              "      <th>pH</th>\n",
              "      <th>EC</th>\n",
              "      <th>%O.C</th>\n",
              "      <th>Aval N</th>\n",
              "      <th>Aval P</th>\n",
              "      <th>Aval K</th>\n",
              "      <th>mg kg</th>\n",
              "      <th>Cu</th>\n",
              "      <th>m..Fe</th>\n",
              "      <th>mg..Mn</th>\n",
              "      <th>S</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Ram Charan</td>\n",
              "      <td>7.5</td>\n",
              "      <td>0.263</td>\n",
              "      <td>0.840</td>\n",
              "      <td>403.768</td>\n",
              "      <td>46.33</td>\n",
              "      <td>793.520</td>\n",
              "      <td>1.03</td>\n",
              "      <td>3.82</td>\n",
              "      <td>26.95</td>\n",
              "      <td>19.19</td>\n",
              "      <td>10.35</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Gajju Patel</td>\n",
              "      <td>7.5</td>\n",
              "      <td>0.286</td>\n",
              "      <td>0.810</td>\n",
              "      <td>389.407</td>\n",
              "      <td>23.46</td>\n",
              "      <td>778.400</td>\n",
              "      <td>1.35</td>\n",
              "      <td>2.75</td>\n",
              "      <td>14.72</td>\n",
              "      <td>16.77</td>\n",
              "      <td>8.28</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Madan Lal</td>\n",
              "      <td>7.2</td>\n",
              "      <td>0.268</td>\n",
              "      <td>0.750</td>\n",
              "      <td>360.685</td>\n",
              "      <td>9.90</td>\n",
              "      <td>554.064</td>\n",
              "      <td>0.60</td>\n",
              "      <td>3.11</td>\n",
              "      <td>15.32</td>\n",
              "      <td>13.27</td>\n",
              "      <td>16.56</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Shiv Prasad</td>\n",
              "      <td>7.5</td>\n",
              "      <td>0.138</td>\n",
              "      <td>0.330</td>\n",
              "      <td>159.631</td>\n",
              "      <td>6.73</td>\n",
              "      <td>214.928</td>\n",
              "      <td>0.28</td>\n",
              "      <td>1.76</td>\n",
              "      <td>12.70</td>\n",
              "      <td>10.85</td>\n",
              "      <td>13.11</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Sunil Patel</td>\n",
              "      <td>7.3</td>\n",
              "      <td>0.170</td>\n",
              "      <td>0.495</td>\n",
              "      <td>238.617</td>\n",
              "      <td>16.23</td>\n",
              "      <td>135.073</td>\n",
              "      <td>0.60</td>\n",
              "      <td>1.40</td>\n",
              "      <td>10.64</td>\n",
              "      <td>10.95</td>\n",
              "      <td>15.18</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>Sohan Dahiya</td>\n",
              "      <td>7.6</td>\n",
              "      <td>0.268</td>\n",
              "      <td>0.420</td>\n",
              "      <td>202.714</td>\n",
              "      <td>2.37</td>\n",
              "      <td>341.600</td>\n",
              "      <td>0.67</td>\n",
              "      <td>3.18</td>\n",
              "      <td>22.13</td>\n",
              "      <td>18.46</td>\n",
              "      <td>8.28</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>Mahendra</td>\n",
              "      <td>7.6</td>\n",
              "      <td>0.152</td>\n",
              "      <td>0.480</td>\n",
              "      <td>231.436</td>\n",
              "      <td>9.50</td>\n",
              "      <td>407.344</td>\n",
              "      <td>0.32</td>\n",
              "      <td>1.41</td>\n",
              "      <td>17.91</td>\n",
              "      <td>8.98</td>\n",
              "      <td>20.01</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>Rampyare</td>\n",
              "      <td>7.4</td>\n",
              "      <td>0.199</td>\n",
              "      <td>0.855</td>\n",
              "      <td>410.949</td>\n",
              "      <td>5.54</td>\n",
              "      <td>165.536</td>\n",
              "      <td>0.60</td>\n",
              "      <td>5.00</td>\n",
              "      <td>24.49</td>\n",
              "      <td>26.39</td>\n",
              "      <td>6.21</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>Sanjay Choudhary</td>\n",
              "      <td>7.1</td>\n",
              "      <td>0.226</td>\n",
              "      <td>0.660</td>\n",
              "      <td>317.602</td>\n",
              "      <td>14.65</td>\n",
              "      <td>334.096</td>\n",
              "      <td>0.71</td>\n",
              "      <td>2.57</td>\n",
              "      <td>16.62</td>\n",
              "      <td>19.18</td>\n",
              "      <td>17.25</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>RajKumar</td>\n",
              "      <td>7.0</td>\n",
              "      <td>0.090</td>\n",
              "      <td>0.450</td>\n",
              "      <td>217.075</td>\n",
              "      <td>7.12</td>\n",
              "      <td>562.240</td>\n",
              "      <td>0.57</td>\n",
              "      <td>2.59</td>\n",
              "      <td>18.38</td>\n",
              "      <td>12.44</td>\n",
              "      <td>5.52</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "               name   pH     EC   %O.C  ...    Cu  m..Fe  mg..Mn      S\n",
              "0        Ram Charan  7.5  0.263  0.840  ...  3.82  26.95   19.19  10.35\n",
              "1       Gajju Patel  7.5  0.286  0.810  ...  2.75  14.72   16.77   8.28\n",
              "2         Madan Lal  7.2  0.268  0.750  ...  3.11  15.32   13.27  16.56\n",
              "3       Shiv Prasad  7.5  0.138  0.330  ...  1.76  12.70   10.85  13.11\n",
              "4       Sunil Patel  7.3  0.170  0.495  ...  1.40  10.64   10.95  15.18\n",
              "5      Sohan Dahiya  7.6  0.268  0.420  ...  3.18  22.13   18.46   8.28\n",
              "6          Mahendra  7.6  0.152  0.480  ...  1.41  17.91    8.98  20.01\n",
              "7          Rampyare  7.4  0.199  0.855  ...  5.00  24.49   26.39   6.21\n",
              "8  Sanjay Choudhary  7.1  0.226  0.660  ...  2.57  16.62   19.18  17.25\n",
              "9          RajKumar  7.0  0.090  0.450  ...  2.59  18.38   12.44   5.52\n",
              "\n",
              "[10 rows x 12 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 157
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mYskuisLxUA4",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import matplotlib.pyplot as plt"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zhiMM4abvJ84",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 612
        },
        "outputId": "fb45872c-9a1f-4b3e-f8e5-9cb409942a55"
      },
      "source": [
        "plt.figure(figsize=(15,10))\n",
        "sns.heatmap(Data.corr(), annot = True)"
      ],
      "execution_count": 159,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7fdd18f0c828>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 159
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\nKSIiIiIiVerPjxX2ArZalrXdsqwy4FNg6D4xpwJ/TEc4bz/bD8uhdkK6A4uNMTuMMTuAJUBP/XK6\niIiIiEj9ZYy51Rizotpf9V86bQrsqra8u2JddWuAKypuXw6EG2MacoQO9XKsQUf6QCIiIiIigq0D\n0y3LegN4o87AA7sXeMUYcwOwENjDUZgF+pA6IQf7dXQRERERETku7QGaV1tuVrGukmVZe6mohBhj\nwoDhlmUd8a9en5C/mC4iIiIiUm/VPVbDLsuBdsaYBLydj2uAa6sHGGNigCzLsjzAg8DbR+OBD3VM\niIiIiIiInEAsy3IB44BZwEbgc8uy1htjnjTGDKkI6w9sMsZsBuKAyUfjsVUJERERERGxkWUd8ZCK\no8ayrOnA9H3WPVrt9pfAl0f7cVUJERERERERW6kSIiIiIiJiJxtnx6qvVAkRERERERFbqRIiIiIi\nImKn+jM7ls+oEiIiIiIiIrZSJURERERExE4aE6JKiIiIiIiI2EuVEBERERERO3nqz++E+IoqISIi\nIiIiYqtjXglJGXzLsX6Ieid+xpu+TsFWYz961tcpiA1K5m70dQq26xZ08n1PE3hJb1+nYLvSWb/4\nOgXbZazy93UKtht2frqvU7DVe9Mb+ToFkYPS5VgiIiIiInbSwHRdjiUiIiIiIvZSJURERERExE76\nsUJVQkRERERExF6qhIiIiIiI2EljQlQJERERERERe6kSIiIiIiJiJ40JUSVERERERETspUqIiIiI\niIidVAlRJUREREREROylSoiIiIiIiI0sy+3rFHxOlRAREREREbGVKiEiIiIiInbSmBBVQkRERERE\nxF6qhIiIiIiI2Em/mK5KiIiIiIiI2EudEBERERERsZUuxxIRERERsZMGpqsSIiIiIiIi9lIlRERE\nRETEThqYfnx2QoLO6knk+DvA4aDwu+nkv/dpje2hV1xK2JVDwePBKioma8rzuBKTCOzVnchxN4O/\nH5S7yHnpX5SuWO2jVhw9E6f8g4WLlhEdFcm3H77u63SOGkfLUwnodxUYB671i3CtmFUrxtmuO/5n\nXgpYeDJ2UzbzbUx4NIGXjgFjwOHEtWYert9+sr8Bh+FkbLN/116E3HInOByUzvmekq8+3n/cWX0J\nf+ApcsffinvrJkx4BGH3P4lf21MonTuTojdetDnzw+fXuSfBN4wDh5Oyud9T+t0n+43z79WX0PFP\n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\n1tgmr8qh2KrWxB4fd5Hf5Jka1a/FvtijHDxyjHPn0pm7Yg2RrXNejJ1MSMRut8Yufvz9PG7tknvM\nn6drFB7E/hNJHDqVZJ2/thyiU/2cWfeoq8OI2W/dZDiZnMq+E4lUD/Inw244lWKNBdp+9DQ7jibQ\nrk7pyOYWF59+NJWoiF5ERfRizuyF9Ol/KwAtWzcjISExV1csm81GcLDVDdHb25uu3aP4Z8t2AOrW\nq5VVrmu3SHbv2uuaSiiPU9AxIb+IyM3GmDlOjeYiTIadv16YQvTXz1hT9E5fyunth2g2/HaOb9jD\nwQVr2TFtKRETH6b3ijdJO5XIsqHvARDS5moaP9oTe3oGxm748/nPST2ZSOrJRPbN/ot/zXsVe3oG\nJzbvY/vUxe6oXv7sds5Oex+/J8YhXl6krZyPPXYfZXreQ8a+7aT/veqim6Yt+Ylyg4bhP9rqqnXu\nj/nYD+1xVeRXzNvLi5GdGzH0u7+w26FX4+rUq1yeSSu30zA0kMj6obSvXZk/9sVz22fLsHnBk50a\nEFTOukA7fDqZuDMptKwRnM9v8hylsc4mw86O5z6h6bRR1hS93ywmedtBaj/TlzMbdnF8Xgzlm9ej\n8Wcj8A7yp1LXltQe0YfVnZ7G76pqXP2fh8BuBy8v9v/3xxyzarmTybDz+4tTuGmqdc7aNn0pJ7cf\nouXw24nfsIf9C9aybdpSIt99mD4r3iT1VCK/Oc5ZJ7cfYvfPf3LnbxOwZ9hZ+cLnGLshft0uds/5\ni9vmWues45v3sXXqYkyGneXPfEL0R09g7HZSTyezbJiHzIwFYM8g+ZN3CRj1Bnh5kbb4V+wH91K2\n72Aydm3jXPYGyQW8GzShbO+7MBkZYLeT/PE7mDOnXRh84XnbbDx/f18eGfMeGXY7vbu0o37Nqrz/\nzc80rFeLqDZNWb1pOxOnzkIQrmtYn1EP9nV32JfN28uLZ7s25ZHp1ni1Xk1rUr9KBSYt20rD8CAi\nrwqnfZ0Q/tgTz20fLcLLS3gqqhFB5XxJTc9gyFfWJA3+ZXwY27Ml3iWgO9qI0eNZve5vTp1KoEvv\nuxl630Bu79kt/w093IJ5S4ju2onVGxaSkpzC40Ofy3pu8YpZREX0okwZX7794RO8fbyx2WwsXfI7\nX3xufbXAfQ/eTafI9pw7l87pU6d59OGR7qqKW+nsWCAFSdOIyBnAH0gFzgECGGNMhUtuCHxR7e5S\n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\nes8YIHy4dFoAAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 1080x720 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "p0ZyxQyql3nW",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "X = Data.drop(['name','Aval N','Aval P','Aval K','mg kg','Cu','m..Fe','mg..Mn','S'], axis =1)\n",
        "y1 = Data['Aval N']\n",
        "y2 = Data['Aval P']\n",
        "y3 = Data['Aval K']"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "f38uhd0Rxs5z",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import tensorflow as tf"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_3sVfu-Lua2X",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "X_train, X_test, y1_train, y1_test = train_test_split(X,y1, test_size = 0.2, random_state = 0)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "EpUJVtYP6EZH",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "y2_train, y2_test = train_test_split(y2, test_size = 0.2, random_state = 0)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "U-_sAGUS6Km4",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "y3_train, y3_test = train_test_split(y3, test_size = 0.2, random_state = 0)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "N20NIOw8yF5A",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "modelA1 = LinearRegression()\n",
        "modelA2 = LinearRegression()\n",
        "modelA3 = LinearRegression()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0JWX38FUyhkd",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "cd3b037f-67cf-45c6-b4c8-7f6271d5dcff"
      },
      "source": [
        "modelA1.fit(X_train, y1_train)"
      ],
      "execution_count": 180,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 180
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Dc6xWi2kE2p4",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "777545e1-1657-4846-fa26-51a4f292a774"
      },
      "source": [
        "modelA2.fit(X_train, y2_train)"
      ],
      "execution_count": 181,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 181
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "HROjYNQeE6Q_",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "3ca3790b-da96-4f98-c485-d811bf8f6413"
      },
      "source": [
        "modelA3.fit(X_train, y3_train)"
      ],
      "execution_count": 182,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 182
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wtpsst3Ny40Y",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "L1_pred = modelA1.predict(X_test)\n",
        "L2_pred = modelA2.predict(X_test)\n",
        "L3_pred = modelA3.predict(X_test)\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "z26eHvgoDRKv",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "c1a682a6-a579-4ef9-95f1-b001655defe5"
      },
      "source": [
        "L1_pred"
      ],
      "execution_count": 177,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([375.66964207, 332.26213321])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 177
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QZRdmjjTBEoI",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "0dc5bf0b-d638-46bd-feb2-ff43fa2849dc"
      },
      "source": [
        "L2_pred"
      ],
      "execution_count": 171,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([25.37942654, 22.06209337])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 171
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "HkdmVdJnDYgg",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "c04e3e7e-b7c3-4914-9411-d085111ca0f0"
      },
      "source": [
        "L3_pred"
      ],
      "execution_count": 178,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([597.89753357, 558.19831197])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 178
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "F4hwZKV13ltU",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "c64104fe-5f1d-4dec-c20d-f311a29c2fce"
      },
      "source": [
        "a= modelA1.score(X_test, y1_test)\n",
        "b= modelA2.score(X_test, y2_test)\n",
        "c= modelA3.score(X_test, y3_test)\n",
        "print('Score for Aval N is {}, Score for Aval P is {}, Score for Aval K is{} '.format(a, b, c))"
      ],
      "execution_count": 184,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Score for Aval N is 0.5264820043653415, Score for Aval P is -25.109852553550727, Score for Aval K is-1.1553056672076427 \n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}