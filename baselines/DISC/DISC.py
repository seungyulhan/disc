import os
import time
import functools
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import set_global_seeds
from baselines.common.policies import build_policy
from baselines.common.runners import AbstractEnvRunner
from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.running_mean_std import RunningMeanStd

from baselines.common.tf_util import initialize

class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, nbatch_act, nsteps, vf_coef):
        sess = get_session()

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            train_model = policy(None, nsteps, sess)

        # CREATE THE PLACEHOLDERS
        A = train_model.pdtype.sample_placeholder([None])
        OLDMEAN = train_model.pdtype.sample_placeholder([None])
        OLDLOGSTD = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        # Cliprange
        EPSILON = tf.placeholder(tf.float32, [])
        # IS target constant
        ALPHA_IS = tf.placeholder(tf.float32, [])
        ON_POLICY_DATA = tf.placeholder(tf.float32, [None])

        neglogpac = train_model.pd.neglogp(A)
        mean = train_model.pd.mean
        logstd = train_model.pd.logstd

        # Calculate the entropy
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - EPSILON, EPSILON)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio of each dimension (pi current policy / pi old policy)
        ratio = tf.exp(- 0.5 * tf.square((A - mean) / tf.exp(logstd)) - logstd + 0.5 * tf.square(
            (A - OLDMEAN) / tf.exp(OLDLOGSTD)) + OLDLOGSTD)
        
        # Compute DISC objective function
        sgn = tf.ones_like(ratio) * tf.expand_dims(tf.sign(ADV), 1)
        ratio_clip = tf.clip_by_value(ratio, 1.0 - EPSILON, 1.0 + EPSILON)
        r = tf.reduce_prod(sgn * tf.minimum(ratio * sgn, ratio_clip * sgn),axis=-1)
        J_DISC = - r * ADV / tf.stop_gradient(tf.reduce_mean(r))

        # Compute IS loss function (Compute IS loss only for on-policy data)
        ISloss = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC) * ON_POLICY_DATA)
        KLloss = tf.reduce_mean(tf.reduce_sum(logstd - OLDLOGSTD + 0.5 * (tf.square(tf.exp(OLDLOGSTD)) \
                + tf.square(mean - OLDMEAN)) / tf.square(tf.exp(logstd)) - 0.5, axis=1) * ON_POLICY_DATA)
        J_DISC = tf.reduce_mean(J_DISC) + ALPHA_IS * ISloss

        # Total loss
        loss = J_DISC + vf_loss * vf_coef

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model')
        print(params)
        # 2. Build our trainer
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = trainer.compute_gradients(loss, params)

        _train = trainer.apply_gradients(grads_and_var)

        def train(lr, epsilon, alpha_IS, obs, returns, advs, actions, values, neglogpacs, old_mean, old_logstd, on_policy_data, rho_now):
            # Normalize the advantages : mean of min(1.0,rho) * A must be zero as described in IMPALA
            radvs = np.minimum(1.0,rho_now)*advs
            advs = (advs - radvs.mean()/np.minimum(1.0,rho_now).mean()) / (radvs.std() + 1e-8)

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    EPSILON:epsilon, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values, OLDMEAN:old_mean, OLDLOGSTD:old_logstd, ALPHA_IS:alpha_IS, ON_POLICY_DATA:on_policy_data}
            return sess.run([J_DISC, vf_loss, entropy, ISloss, KLloss, _train],td_map)[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'ISloss', 'KLloss']

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.meanlogstd = act_model.meanlogstd
        self.value = act_model.value
        self.values = train_model.value
        self.meanlogstds = train_model.meanlogstd
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()

class Runner(AbstractEnvRunner):
    """
    We use this object to make a sample batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a sample batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.clipob = 10.
        self.cliprew = 10.
        self.eps = 1e-8
        self.ret = 0
        # Filter for normalized observation and return
        self.ob_rms = RunningMeanStd(shape=self.env.observation_space.shape)
        self.ret_rms = RunningMeanStd(shape=())

    def obfilt(self, obs):
        obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.eps), -self.clipob,
                      self.clipob)
        return obs

    def rewfilt(self, rews):
        rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.eps), -self.cliprew, self.cliprew)
        return rews

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_dones, mb_neglogpacs , mb_means , mb_logstds = [],[],[],[],[],[],[]
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action and neglopacs
            actions, _, _, neglogpacs = self.model.step(self.obfilt(self.obs))
            means, logstds = self.model.meanlogstd(self.obfilt(self.obs))
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            mb_means.append(means)
            mb_logstds.append(logstds)

            # Take actions in env and look the results
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            self.ob_rms.update(self.obs)
            self.ret =  self.ret * self.gamma + rewards
            self.ret_rms.update(self.ret)
            if self.dones:
                self.ret = 0

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_means = np.asarray(mb_means)
        mb_logstds = np.asarray(mb_logstds)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        return (*map(sf01, (mb_obs, mb_rewards, mb_dones, mb_actions, mb_neglogpacs, mb_means, mb_logstds)),
            self.obs, self.dones, epinfos)

class EvalRunner(AbstractEnvRunner):
    """
    We use this object to make a sample batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a sample batch
    """

    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.obfilt = None
        self.rewfilt = None

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        epinfos = []
        # For n in range number of steps
        epi_lens=0
        for _ in range(self.nsteps):
            # Run deterministic evaluation steps
            actions, logstds = self.model.meanlogstd(self.obfilt(self.obs))

            # Take actions in env and look the results
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            if self.dones:
                epi_lens+=1
            if epi_lens==10:
                break
        return epinfos

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

def learn(*, network, env, total_timesteps, nsteps=2048, lr=3e-4, vf_coef=0.5, gamma=0.99, lam=0.95, log_interval=1,
          save_interval=0, load_path=None, gradstepsperepoch=32, noptepochs=10, epsilon=0.4, replay_length=64,
          J_targ=0.001, epsilon_b=0.1, gaev = 1, eval_env = None, seed=None,


            **network_kwargs):
    '''
    Dimension-Wise Importance Sampling Weight Clipping (DISC) parameters

    Parameters:
    ----------

    network:                          multi-layer perceptrons (MLP) with 2 hidden layers of size 64

    env:                              Mujoco environment

    eval_env:                         environment for the deterministic evaluation

    total_timesteps: int              number of time steps

    nsteps (N): int                   size of a sample batch

    lr (beta): float function         learning rate which reduces linearly as iterations goes on

    vf_coef: float                    value function loss coefficient

    gamma: float                      discounting factor

    lam (lambda) : float              discounting factor for GAE

    log_interval: int                 number of time steps between logging events

    save_interval: int                number of time steps between saving events

    load_path: str                    path to load the model from

    gradstepsperepoch: int            number of training per epoch

    noptepochs: int                   number of training epochs per update

    epsilon : float                   clipping factor for dimension-wise clipping

    replay length (L) : int           maximum number of sample batches stored in the replay buffer

    J_targ: float                     IS target constant

    epsilon_b : float                 batch inclusion factor

    gaev : int                        use GAE-V if gaev = 1, and use GAE otherwise

    '''

    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(epsilon, float): epsilon = constfn(epsilon)
    else: assert callable(epsilon)
    total_timesteps = int(total_timesteps)

    policy = build_policy(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space
    obdim = ob_space.shape[0]
    acdim = ac_space.shape[0]
    print("Observation space dimension : " + str(obdim))
    print("Action space dimension : " + str(acdim))

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // gradstepsperepoch

    # Instantiate the model object (that creates act_model and train_model)
    make_model = lambda : Model(policy=policy, nbatch_act=nenvs, nsteps=nsteps, vf_coef=vf_coef)
    model = make_model()
    if load_path is not None:
        model.load(load_path)
    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    if eval_env is not None:
        eval_runner = EvalRunner(env = eval_env, model = model, nsteps = 10*nsteps, gamma = gamma, lam= lam)
        eval_runner.obfilt=runner.obfilt
        eval_runner.rewfilt=runner.rewfilt

    epinfobuf = deque(maxlen=10)

    # Start total timer
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch

    def GAE(seg, gamma, value, lam):
        """
        Compute target value using TD(lambda) estimator, and advantage with GAE
        """
        done = np.append(seg["done"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1

        T = len(seg["rew"])
        gaelam = np.empty(T, 'float32')
        rew = runner.rewfilt(seg["rew"])
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - done[t + 1]
            delta = rew[t] + gamma * value[t + 1] * nonterminal - value[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        ret = gaelam + value[:-1]
        return ret, gaelam

    def GAE_V(seg, gamma, value, rho):
        """
        Compute target value using V-trace estimator, and advantage with GAE-V
        """
        done = np.append(seg["done"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
        rho_ = np.append(rho, 1.0)
        r = np.minimum(1.0, rho_)

        T = len(seg["rew"])
        gaelam = np.empty(T, 'float32')
        rew = runner.rewfilt(seg["rew"])
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - done[t + 1]
            delta = (rew[t] + gamma * value[t + 1] * nonterminal - value[t])
            gaelam[t] = delta + gamma * lam * nonterminal * lastgaelam
            lastgaelam = r[t] * gaelam[t]
        ret = r[:-1]*gaelam + value[:-1]
        return ret, gaelam

    seg = None
    # Calculate the epsilon
    epsilonnow = epsilon(1.0)
    alpha_IS=1.0
    for update in range(1, nupdates+1):
        assert nbatch % gradstepsperepoch == 0
        # Start timer
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = np.maximum(1e-4, lr(frac))

        if seg is None:
            prev_seg = seg
            seg = {}
        else:
            prev_seg = {}
            for i in seg:
                prev_seg[i] = np.copy(seg[i])

        # Run a sample batch
        seg["ob"], seg["rew"], seg["done"], seg["ac"], seg["neglogp"], seg["mean"], seg["logstd"], final_obs, final_done, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfos = eval_runner.run()

        # Stack the sample batches (the maximum length is L)
        if prev_seg is not None:
            for key in seg:
                if len(np.shape(seg[key])) == 1:
                    seg[key] = np.hstack([prev_seg[key], seg[key]])
                else:
                    seg[key] = np.vstack([prev_seg[key], seg[key]])
                if np.shape(seg[key])[0] > replay_length * nsteps:
                    seg[key] = seg[key][-replay_length * nsteps:]

        # Compute all values of all samples in the buffer
        ob_stack = np.vstack([seg["ob"], final_obs])
        values = model.values(runner.obfilt(ob_stack))
        values[-1] = (1.0-final_done) * values[-1]
        ob = runner.obfilt(seg["ob"])

        # Compute IS weight of all samples in the buffer
        mean_now, logstd_now = model.meanlogstds(ob)
        neglogpnow = 0.5 * np.sum(np.square((seg["ac"] - mean_now) / np.exp(logstd_now)), axis=-1) \
                      + 0.5 * np.log(2.0 * np.pi) * np.shape(seg["ac"])[1] \
                      + np.sum(logstd_now, axis=-1)
        rho = np.exp(-neglogpnow + seg["neglogp"])

        # Estimate target values and advantages
        if gaev==1:
            ret, gae = GAE_V(seg, gamma, values, rho)
        else:
            ret, gae = GAE(seg, gamma, values, lam)

        # Select sample batches which satisfies batch limiting condition in the paper
        prior_prob = np.zeros(len(seg["ob"]))
        rho_dim =  np.exp(- 0.5 * np.square((seg["ac"] - mean_now) / np.exp(logstd_now)) \
                - logstd_now + 0.5 * np.square((seg["ac"] - seg["mean"]) / np.exp(seg["logstd"])) + seg["logstd"])

        for i in range(int(len(prior_prob) / nsteps)):
            batch_condition = np.mean(np.abs(rho_dim[i * nsteps:(i + 1) * nsteps] - 1.0) + 1.0)
            if batch_condition > 1 + epsilon_b:
                prior_prob[i * nsteps:(i + 1) * nsteps] = 0
            else:
                prior_prob[i * nsteps:(i + 1) * nsteps] = 1



        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        # Index of each element of batch_size
        # Create the indices array

        # On-policy data indices and minibatch size
        inds_on = np.arange(nsteps)+len(seg["ob"]) - nsteps
        nbatch_adapt_on = int((nsteps) / nsteps * nbatch_train)

        # Off-policy data indices and minibatch size
        inds_off = np.arange(len(seg["ob"]) - nsteps)
        nbatch_adapt_off = int((np.sum(prior_prob) - nsteps) / nsteps * nbatch_train)
        
        # On-policy data index
        on_policy_data = np.ones(len(seg["ob"])) * np.sum(prior_prob) / nsteps
        on_policy_data[:-nsteps]=0

        for _ in range(noptepochs):
            losses_epoch = []
            for _ in range(int(nsteps/nbatch_train)):
                # Choose sample minibatch indices of off policy trajectories
                if nbatch_adapt_off>0:
                    idx_off = np.random.choice(inds_off, nbatch_adapt_off,p=prior_prob[:-nsteps]/np.sum(prior_prob[:-nsteps]))
                else:
                    idx_off = []

                # Choose sample minibatch indices of on policy trajectories
                idx_on = np.random.choice(inds_on, nbatch_adapt_on)

                all_idx = np.hstack([idx_off,idx_on]).astype(int)

                # Sample minibatch
                slices = (arr[all_idx] for arr in (ob, ret, gae, seg["ac"], values[:-1], seg["neglogp"], seg["mean"], seg["logstd"], on_policy_data, rho))

                # Train the model
                loss_epoch = model.train(lrnow, epsilonnow, alpha_IS, *slices)
                mblossvals.append(loss_epoch)
                losses_epoch.append(loss_epoch)

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)

        # Update adaptive IS target constant
        print("IS loss avg :", lossvals[3])
        if lossvals[3] > J_targ * 1.5:
            alpha_IS *= 2

            print("Adaptive IS loss factor is increased")
        elif lossvals[3] < J_targ / 1.5:
            alpha_IS /= 2
            print("Adaptive IS loss factor is reduced")
        alpha_IS = np.clip(alpha_IS,2**(-10),64)

        # End timer
        tnow = time.time()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            logger.logkv("adaptive IS loss factor", alpha_IS)
            logger.logkv("clipping factor", epsilonnow)
            logger.logkv("learning rate", lrnow)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfos]))
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfos]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
    return model

# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)



