package main

import (
	"bufio"
	"encoding/json"
	"io"
	"os"

	log "github.com/Sirupsen/logrus"
	"github.com/haruyama/adam"
)

type input struct {
	Data  map[string]float64 `json:"data"`
	Label float64            `json:"label"`
}

func main() {
	log.SetLevel(log.DebugLevel)
	log.Debug("Start")

	adam := adam.NewAdam()
	bio := bufio.NewReader(os.Stdin)

	for {
		line, prefix, err := bio.ReadLine()
		if prefix {
			log.Fatal("long line not supported")
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}

		var in input
		if err := json.Unmarshal(line, &in); err != nil {
			log.Fatal(err)
		}
		if in.Label == 0 {
			in.Label = float64(adam.Classify(in.Data))
			log.Debugf("Classify: %v", in)
		} else {
			adam.Update(int(in.Label), in.Data)
			log.Debugf("Update: %v", in)
		}
	}
	log.Debug("Weight: %v", adam.Weight)
}
