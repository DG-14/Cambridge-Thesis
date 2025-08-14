reward_scaling_stats = {
  "0": {
    "terminal_reward": {
      "mean": 2.415323488061946,
      "std": 2.483024312745438
    },
    "step_reward": {
      "mean": 8.854361307585718e-05,
      "std": 1.3070474541588625e-05
    }
  },
  "1": {
    "terminal_reward": {
      "mean": 2.3928276873955836,
      "std": 2.6790810439535813
    },
    "step_reward": {
      "mean": -0.04892866045861766,
      "std": 0.07573948131540974
    }
  },
  "2": {
    "terminal_reward": {
      "mean": 1.0853902616345081,
      "std": 2.0714725035937698
    },
    "step_reward": {
      "mean": -0.06351519444133706,
      "std": 0.07363476474505402
    }
  },
  "3": {
    "terminal_reward": {
      "mean": -0.10008032367721899,
      "std": 0.904209281207031
    },
    "step_reward": {
      "mean": -0.12482889137603703,
      "std": 0.09474206360605691
    }
  },
  "4": {
    "terminal_reward": {
      "mean": -0.3418407162697691,
      "std": 0.22152484175480153
    },
    "step_reward": {
      "mean": -0.14162761868296378,
      "std": 0.07069139346250797
    }
  },
  "5": {
    "terminal_reward": {
      "mean": -0.3334387779407426,
      "std": 0.35408706311876126
    },
    "step_reward": {
      "mean": -0.13648212468047485,
      "std": 0.0823653977893553
    }
  }
}

def normalise_step(step_reward,difficulty):

    # scaled_reward = (step_reward - reward_scaling_stats[str(int(difficulty))]["step_reward"]["mean"]) / (reward_scaling_stats[str(int(difficulty))]["step_reward"]["std"] + 1e-8)


    # return scaled_reward

    return step_reward

def normalise_terminal(terminal_reward,difficulty):

    # scaled_reward = (terminal_reward - reward_scaling_stats[str(int(difficulty))]["terminal_reward"]["mean"]) / (reward_scaling_stats[str(int(difficulty))]["terminal_reward"]["std"] + 1e-8)


    # return scaled_reward

    return terminal_reward
# reward_scaling_stats = {
#   "0": {
#     "terminal_reward": {
#       "mean": 0,
#       "std": 1
#     },
#     "step_reward": {
#       "mean": 0,
#       "std": 1e-05
#     }
#   },
#   "1": {
#     "terminal_reward": {
#       "mean": 0,
#       "std": 1
#     },
#     "step_reward": {
#       "mean": 0,
#       "std": 1e-05
#     }
#   },
#   "2": {
#     "terminal_reward": {
#       "mean": 0,
#       "std": 1
#     },
#     "step_reward": {
#       "mean": 0,
#       "std": 1e-05
#     }
#   },
#   "3": {
#     "terminal_reward": {
#       "mean": 0,
#       "std": 1
#     },
#     "step_reward": {
#       "mean": 0,
#       "std": 1e-05
#     }
#   },
#   "4": {
#     "terminal_reward": {
#       "mean": 0,
#       "std": 1
#     },
#     "step_reward": {
#       "mean": 0,
#       "std": 1e-05
#     }
#   },
#   "5": {
#     "terminal_reward": {
#       "mean": 0,
#       "std": 1
#     },
#     "step_reward": {
#       "mean": 0,
#       "std": 1e-05
#     }
#   }
# }
