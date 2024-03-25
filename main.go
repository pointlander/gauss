// Copyright 2024 The Gauss Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"github.com/pointlander/datum/iris"
	. "github.com/pointlander/matrix"
)

func main() {
	rng := Rand(1)
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	points := NewMatrix(4, 150)
	for _, row := range datum.Fisher {
		for _, value := range row.Measures {
			points.Data = append(points.Data, float32(value))
		}
	}
	multi := NewMultiFromData(points.T())
	multi.LearnA(&rng, nil)
	fmt.Println(multi.E)
	fmt.Println(multi.A)
}
