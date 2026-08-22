package config

import "flag"

// FlagOverrides holds the destinations for CLI flags that can override a
// loaded Settings. All numeric flags default to their zero value; a flag
// only takes effect if the user actually passed it (see Apply), so a
// zero value typed by the user (e.g. "-seed=0") is still honored while an
// un-passed flag never clobbers the config file's value.
type FlagOverrides struct {
	ConfigPath string

	Epochs       int
	RolloutSize  int
	EpisodeLen   int
	Gamma        float64
	LearningRate float64
	GridSize     int
	HiddenSize   int
	Seed         uint64
	Workers      int
}

// RegisterFlags registers every Settings field as a flag on fs and
// returns the destination struct Apply needs to overlay them.
func RegisterFlags(fs *flag.FlagSet) *FlagOverrides {
	o := &FlagOverrides{}

	fs.StringVar(&o.ConfigPath, "config", "configs/config.json", "path to a JSON config file")
	fs.IntVar(&o.Epochs, "epochs", 0, "number of training epochs (overrides config file)")
	fs.IntVar(&o.RolloutSize, "rollout-size", 0, "trajectories collected per epoch (overrides config file)")
	fs.IntVar(&o.EpisodeLen, "episode-len", 0, "maximum steps per episode (overrides config file)")
	fs.Float64Var(&o.Gamma, "gamma", 0, "discount factor (overrides config file)")
	fs.Float64Var(&o.LearningRate, "learning-rate", 0, "SGD learning rate (overrides config file)")
	fs.IntVar(&o.GridSize, "grid-size", 0, "number of grid cells, must be a perfect square (overrides config file)")
	fs.IntVar(&o.HiddenSize, "hidden-size", 0, "hidden layer width (overrides config file)")
	fs.Uint64Var(&o.Seed, "seed", 0, "master RNG seed (overrides config file)")
	fs.IntVar(&o.Workers, "workers", 0, "number of concurrent rollout workers (overrides config file)")

	return o
}

// Apply overlays onto settings only the flags that were explicitly passed
// on the command line (per fs.Visit, which only visits flags that were
// set), implementing "CLI arguments override config file values when both
// are present" without a zero-valued flag being mistaken for "unset".
func Apply(settings *Settings, fs *flag.FlagSet, o *FlagOverrides) {
	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "epochs":
			settings.Epochs = o.Epochs
		case "rollout-size":
			settings.RolloutSize = o.RolloutSize
		case "episode-len":
			settings.EpisodeLen = o.EpisodeLen
		case "gamma":
			settings.Gamma = float32(o.Gamma)
		case "learning-rate":
			settings.LearningRate = float32(o.LearningRate)
		case "grid-size":
			settings.GridSize = o.GridSize
		case "hidden-size":
			settings.HiddenSize = o.HiddenSize
		case "seed":
			settings.Seed = o.Seed
		case "workers":
			settings.Workers = o.Workers
		}
	})
}
