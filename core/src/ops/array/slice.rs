use crate::internal::*;
use crate::num_traits::Zero;

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Slice {
    pub axis: usize,
    pub start: TDim,
    pub end: TDim,
    pub stride: isize, // can be negative to revert iteration order ...
}

impl DynHash for Slice {
    fn dyn_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        dyn_hash(&self, hasher)
    }
}

impl Slice {
    pub fn new(axis: usize, start: impl ToDim, end: impl ToDim, stride: isize) -> Slice {
        Slice { axis, start: start.to_dim(), end: end.to_dim(), stride }
    }

    pub fn suffix(&self, name: &str) -> String {
        // show stride
        let stride_suffix =
            if self.stride == 1 { String::from("") } else { format!("__stride_{}", self.stride) };
        format!("{}.axis{}_{}_{}{}", name, self.axis, self.start, self.end, stride_suffix)
    }
}

impl Op for Slice {
    fn name(&self) -> Cow<str> {
        "Slice".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        // show stride
        let stride_info =
            if self.stride == 1 { String::from("") } else { format!(", stride: {}", self.stride) };

        Ok(vec![format!("axis: {}, {}..{}{}", self.axis, self.start, self.end, stride_info)])
    }

    op_as_typed_op!();

    fn same_as(&self, other: &dyn Op) -> bool {
        if let Some(other) = other.downcast_ref::<Self>() {
            other == self
        } else {
            false
        }
    }
}

impl EvalOp for Slice {
    fn is_stateless(&self) -> bool {
        self.start.to_usize().is_ok() && self.end.to_usize().is_ok()
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let start = self.start.to_usize()?;
        let end = self.end.to_usize()?;
        eval_slice(&input, self.axis, start, end, self.stride)
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(if !self.is_stateless() { Some(Box::new(self.clone())) } else { None })
    }
}

impl OpState for Slice {
    fn eval(
        &mut self,
        session: &mut SessionState,
        _op: &dyn Op,
        mut inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let start = self.start.eval(&session.resolved_symbols).to_usize()?;
        let end = self.end.eval(&session.resolved_symbols).to_usize()?;
        eval_slice(&input, self.axis, start, end, self.stride)
    }
}

fn eval_slice(
    input: &Tensor,
    axis: usize,
    start: usize,
    end: usize,
    stride: isize,
) -> TractResult<TVec<Arc<Tensor>>> {
    if end > input.shape()[axis] || start > end {
        bail!("Invalid range {}..{} for slicing {:?} on axis {}", start, end, input, axis);
    }
    if stride < 0 || (end as isize) < stride {
        bail!("Invalid stride {} for slicing {:?} on axis {}", stride, input, axis);
    }
    unsafe {
        let mut shape: TVec<_> = input.shape().into();
        shape[axis] = end - start;
        let mut tensor = Tensor::uninitialized_dt(input.datum_type(), &shape)?;
        tensor.assign_slice_with_stride_unchecked(.., input, start..end, stride, axis);
        Ok(tvec!(tensor.into_arc_tensor()))
    }
}

impl TypedOp for Slice {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let n_collected_values = ((self.end.clone() - &self.start).to_i64()? as f32
            / self.stride.abs() as f32)
            .ceil() as i64;
        let mut new_fact = inputs[0].datum_type.fact(&*inputs[0].shape);
        new_fact.shape.set(self.axis, n_collected_values.to_dim());
        Ok(tvec!(new_fact))
    }

    fn invariants(
        &self,
        inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<Invariants> {
        let axes =
            (0..inputs[0].rank()).filter(|&ax| self.axis != ax).map(AxisInfo::simple).collect();
        Ok(axes)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        if let Some(axis) = change.transform_axis(self.axis) {
            if axis != self.axis {
                Ok(Some(AxisChangeConsequence::new(
                    model,
                    node,
                    Some(Box::new(Slice { axis, ..self.clone() }) as _),
                    change,
                )))
            } else {
                Ok(Some(AxisChangeConsequence::new(model, node, None, change)))
            }
        } else {
            Ok(None)
        }
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let prec = model.node(node.inputs[0].node);
        if self.start.is_zero()
            && (self.end == model.outlet_fact(node.inputs[0])?.shape[self.axis])
            && self.stride == 1
        {
            return Ok(Some(TypedModelPatch::shunt_one_op(model, node)?.with_context("noop")));
        }
        let (start, end) = if let (Ok(s), Ok(e)) = (self.start.to_usize(), self.end.to_usize()) {
            (s, e)
        } else {
            return Ok(None);
        };
        dbg!("HELLLLLLLLLLLLOOOOOOOOOOOOOOOOOOO");
        let mut patch = TypedModelPatch::default();

        if let Some((wire, no_slice_op)) = prec.op().as_typed().unwrap().slice_output(
            model,
            prec,
            &mut patch,
            &self.suffix(&node.name),
            node.inputs[0].slot,
            self.axis,
            start,
            end,
            self.stride,
        )? {
            /*
            dbg!(node);
            dbg!(prec);
            dbg!(&patch);
            dbg!(no_slice_op);
            */
            if !no_slice_op {
                dbg!("I DID no_slice_op IN DECLUTTER");
                return Ok(None);
            }
            patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
            dbg!("I DID patch.shunt_outside IN DECLUTTER");
            return Ok(Some(patch));
        }
        dbg!("I AM AT THE END OF DECLUTTER");
        Ok(None)
    }

    fn slice_output(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        patch: &mut TypedModelPatch,
        suffix: &str,
        _output_slot: usize,
        axis: usize,
        start: usize,
        end: usize,
        stride: isize,
    ) -> TractResult<Option<(OutletId, bool)>> {
        let prec = model.node(node.inputs[0].node);
        if axis != self.axis {
            let suffix = self.suffix(&node.name) + "." + suffix;
            return prec
                .op()
                .as_typed()
                .unwrap()
                .slice_output(
                    model,
                    prec,
                    patch,
                    &suffix,
                    node.inputs[0].slot,
                    axis,
                    start,
                    end,
                    stride,
                )?
                .map(|(w, no_slice_op)| {
                    Ok((
                        patch.wire_node(
                            format!("{}.{}", node.name, &suffix),
                            self.clone(),
                            &[w],
                        )?[0],
                        no_slice_op,
                    ))
                })
                .transpose();
        }
        Ok(None)
    }

    fn concretize_dims(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        values: &SymbolValues,
    ) -> TractResult<TVec<OutletId>> {
        let op = Slice {
            axis: self.axis,
            start: self.start.eval(values),
            end: self.end.eval(values),
            stride: self.stride,
        };
        let inputs = node.inputs.iter().map(|i| mapping[i]).collect::<TVec<_>>();
        target.wire_node(&node.name, op, &inputs)
    }

    as_op!();
}
