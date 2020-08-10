// This file is part of Substrate.

// Copyright (C) 2020 Parity Technologies (UK) Ltd.
// SPDX-License-Identifier: Apache-2.0

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Benchmarks for the contracts pallet

#![cfg(feature = "runtime-benchmarks")]

use crate::*;
use crate::Module as Contracts;

use frame_benchmarking::{benchmarks, account};
use frame_system::{Module as System, RawOrigin};
use parity_wasm::elements::{Instruction, Instructions, FuncBody, ValueType};
use sp_runtime::traits::{Hash, Bounded, SaturatedConversion, CheckedDiv};
use sp_std::{default::Default, convert::TryFrom};

/// How many batches we do per API benchmark. 
const API_BENCHMARK_BATCHES: u32 = 20;

/// How many API calls are executed in a single batch. The reason for increasing the amount
/// of API calls in batches (per benchmark component increase) is so that the linear regression
/// has an easier time determining the contribution of that component.
const API_BENCHMARK_BATCH_SIZE: u32 = 100;

struct WasmModule<T:Trait> {
	code: Vec<u8>,
	hash: <T::Hashing as Hash>::Output,
}

struct Contract<T: Trait> {
	caller: T::AccountId,
	account_id: T::AccountId,
	addr: <T::Lookup as StaticLookup>::Source,
	endowment: BalanceOf<T>,
}

struct ModuleDefinition {
	data_segments: Vec<DataSegment>,
	memory: Option<ImportedMemory>,
	imported_functions: Vec<ImportedFunction>,
	deploy_body: Option<FuncBody>,
	call_body: Option<FuncBody>,
}

impl Default for ModuleDefinition {
	fn default() -> Self {
		Self {
			data_segments: vec![],
			memory: None,
			imported_functions: vec![],
			deploy_body: None,
			call_body: None,
		}
	}
}

struct ImportedFunction {
	name: &'static str,
	params: Vec<ValueType>,
	return_type: Option<ValueType>,
}

struct ImportedMemory {
	min_pages: u32,
	max_pages: u32,
}

struct DataSegment {
	offset: u32,
	value: Vec<u8>,
}

fn create_code<T: Trait>(def: ModuleDefinition) -> WasmModule<T> {
	// internal functions start at that offset.
	let func_offset = u32::try_from(def.imported_functions.len()).unwrap();

	// Every contract must export "deploy" and "call" functions
	let mut contract = parity_wasm::builder::module()
		// deploy function (first internal function)
		.function()
			.signature().with_params(vec![]).with_return_type(None).build()
			.with_body(def.deploy_body.unwrap_or_else(||
				FuncBody::new(Vec::new(), Instructions::empty())
			))
			.build()
		// call function (second internal function)
		.function()
			.signature().with_params(vec![]).with_return_type(None).build()
			.with_body(def.call_body.unwrap_or_else(||
				FuncBody::new(Vec::new(), Instructions::empty())
			))
			.build()
		.export().field("deploy").internal().func(func_offset).build()
		.export().field("call").internal().func(func_offset + 1).build();

	// Grant access to linear memory.
	if let Some(memory) = def.memory {
		contract = contract.import()
			.module("env").field("memory")
			.external().memory(memory.min_pages, Some(memory.max_pages))
			.build();
	}

	// Import supervisor functions. They start with idx 0.
	for func in def.imported_functions {
		let sig = parity_wasm::builder::signature()
			.with_params(func.params)
			.with_return_type(func.return_type)
			.build_sig();
		let sig = contract.push_signature(sig);
		contract = contract.import()
			.module("seal0")
			.field(func.name)
			.with_external(parity_wasm::elements::External::Function(sig))
			.build();
	}

	// Initialize memory
	for data in def.data_segments {
		contract = contract.data()
			.offset(Instruction::I32Const(data.offset as i32))
			.value(data.value)
			.build()
	}

	let code = contract.build().to_bytes().unwrap();
	let hash = T::Hashing::hash(&code);
	WasmModule {
		code,
		hash
	}
}

fn body(instructions: Vec<Instruction>) -> FuncBody {
	FuncBody::new(Vec::new(), Instructions::new(instructions))
}

fn body_from_repeated(instructions: &[Instruction], repetitions: u32) -> FuncBody {
	let instructions = Instructions::new(
		instructions
			.iter()
			.cycle()
			.take(instructions.len() * usize::try_from(repetitions).unwrap())
			.cloned()
			.chain(sp_std::iter::once(Instruction::End))
			.collect()
	);
	FuncBody::new(Vec::new(), instructions)
}

fn dummy_code<T: Trait>() -> WasmModule<T> {
	create_code::<T>(Default::default())
}

fn sized_code<T: Trait>(target_bytes: u32) -> WasmModule<T> {
	use parity_wasm::elements::{
		Instruction::{If, I32Const, Return, End},
		BlockType
	};
	// Base size of a contract is 47 bytes and each expansion adds 6 bytes.
	// We do one expansion less to account for the code section and function body
	// size fields inside the binary wasm module representation which are leb128 encoded
	// and therefore grow in size when the contract grows. We are not allowed to overshoot
	// because of the maximum code size that is enforced by `put_code`.
	let expansions = (target_bytes.saturating_sub(47) / 6).saturating_sub(1);
	const EXPANSION: [Instruction; 4] = [
		I32Const(0),
		If(BlockType::NoResult),
		Return,
		End,
	];
	create_code::<T>(ModuleDefinition {
		call_body: Some(body_from_repeated(&EXPANSION, expansions)),
		.. Default::default()
	})
}

fn getter_code<T: Trait>(getter_name: &'static str, repeat: u32) -> WasmModule<T> {
	let pages = max_pages::<T>();
	create_code::<T>(ModuleDefinition {
		memory: Some(ImportedMemory { min_pages: pages, max_pages: pages }),
		imported_functions: vec![ImportedFunction {
			name: getter_name,
			params: vec![ValueType::I32, ValueType::I32],
			return_type: None,
		}],
		// Write the output buffer size. The output size will be overwritten by the
		// supervisor with the real size when calling the getter. Since this size does not
		// change between calls it suffices to start with an initial value and then just
		// leave as whatever value was written there.
		data_segments: vec![DataSegment {
			offset: 0,
			value: (pages * 64 * 1024 - 4).to_le_bytes().to_vec(),
		}],
		call_body: Some(body_from_repeated(&[
			Instruction::I32Const(4), // ptr where to store output
			Instruction::I32Const(0), // ptr to length
			Instruction::Call(0), // call the imported function
		], repeat)),
		.. Default::default()
	})
}

fn instantiate_contract<T: Trait>(
	module: WasmModule<T>,
	data: Vec<u8>,
) -> Result<Contract<T>, &'static str>
{
	// storage_size cannot be zero because otherwise a contract that is just above the subsistence
	// threshold does not pay rent given a large enough subsistence threshold. But we need rent
	// payments to occur in order to benchmark for worst cases.
	let storage_size = Config::<T>::subsistence_threshold_uncached()
		.checked_div(&T::RentDepositOffset::get())
		.unwrap_or_else(Zero::zero);

	// Endowment should be large but not as large to inhibit rent payments.
	let endowment = T::RentDepositOffset::get()
		.saturating_mul(storage_size + T::StorageSizeOffset::get().into())
		.saturating_sub(1.into());

	let caller = create_funded_user::<T>("instantiator", 0);
	let addr = T::DetermineContractAddress::contract_address_for(&module.hash, &data, &caller);
	init_block_number::<T>();
	Contracts::<T>::put_code_raw(module.code)?;
	Contracts::<T>::instantiate(
		RawOrigin::Signed(caller.clone()).into(),
		endowment,
		Weight::max_value(),
		module.hash,
		data,
	)?;
	let mut contract = get_alive::<T>(&addr)?;
	contract.storage_size = storage_size.saturated_into::<u32>();
	ContractInfoOf::<T>::insert(&addr, ContractInfo::Alive(contract));
	Ok(Contract {
		caller,
		account_id: addr.clone(),
		addr: T::Lookup::unlookup(addr),
		endowment,
	})
}

fn get_alive<T: Trait>(addr: &T::AccountId) -> Result<AliveContractInfo<T>, &'static str> {
	ContractInfoOf::<T>::get(&addr).and_then(|c| c.get_alive())
		.ok_or("Expected contract to be alive at this point.")
}

fn ensure_alive<T: Trait>(addr: &T::AccountId) -> Result<(), &'static str> {
	get_alive::<T>(addr).map(|_| ())
}

fn ensure_tombstone<T: Trait>(addr: &T::AccountId) -> Result<(), &'static str> {
	ContractInfoOf::<T>::get(&addr).and_then(|c| c.get_tombstone())
		.ok_or("Expected contract to be a tombstone at this point.")
		.map(|_| ())
}

fn max_pages<T: Trait>() -> u32 {
	Contracts::<T>::current_schedule().max_memory_pages
}

fn funding<T: Trait>() -> BalanceOf<T> {
	BalanceOf::<T>::max_value() / 2.into()
}

fn create_funded_user<T: Trait>(string: &'static str, n: u32) -> T::AccountId {
	let user = account(string, n, 0);
	T::Currency::make_free_balance_be(&user, funding::<T>());
	user
}

fn eviction_at<T: Trait>(addr: &T::AccountId) -> Result<T::BlockNumber, &'static str> {
	match crate::rent::compute_rent_projection::<T>(addr).map_err(|_| "Invalid acc for rent")? {
		RentProjection::EvictionAt(at) => Ok(at),
		_ => Err("Account does not pay rent.")?,
	}
}

/// Set the block number to one.
///
/// The default block number is zero. The benchmarking system bumps the block number
/// to one for the benchmarking closure when it is set to zero. In order to prevent this
/// undesired implicit bump (which messes with rent collection), wo do the bump ourselfs
/// in the setup closure so that both the instantiate and subsequent call are run with the
/// same block number.
fn init_block_number<T: Trait>() {
	System::<T>::set_block_number(1.into());
}

benchmarks! {
	_ {
	}

	// This extrinsic is pretty much constant as it is only a simple setter.
	update_schedule {
		let schedule = Schedule {
			version: 1,
			.. Default::default()
		};
	}: _(RawOrigin::Root, schedule)

	// This constructs a contract that is maximal expensive to instrument.
	// It creates a maximum number of metering blocks per byte.
	put_code {
		let n in 0 .. Contracts::<T>::current_schedule().max_code_size;
		let caller = create_funded_user::<T>("caller", 0);
		let module = sized_code::<T>(n);
		let origin = RawOrigin::Signed(caller);
	}: _(origin, module.code)

	// Instantiate uses a dummy contract constructor to measure the overhead of the instantiate.
	// The size of the input data influences the runtime because it is hashed in order to determine
	// the contract address.
	instantiate {
		let n in 0 .. max_pages::<T>() * 64;
		let data = vec![42u8; (n * 1024) as usize];
		let endowment = Config::<T>::subsistence_threshold_uncached();
		let caller = create_funded_user::<T>("caller", 0);
		let WasmModule { code, hash } = dummy_code::<T>();
		let origin = RawOrigin::Signed(caller.clone());
		let addr = T::DetermineContractAddress::contract_address_for(&hash, &data, &caller);
		Contracts::<T>::put_code_raw(code)?;
	}: _(origin, endowment, Weight::max_value(), hash, data)
	verify {
		// endowment was removed from the caller
		assert_eq!(T::Currency::free_balance(&caller), funding::<T>() - endowment);
		// contract has the full endowment because no rent collection happended
		assert_eq!(T::Currency::free_balance(&addr), endowment);
		// instantiate should leave a alive contract
		ensure_alive::<T>(&addr)?;
	}

	// We just call a dummy contract to measure to overhead of the call extrinsic.
	// The size of the data has no influence on the costs of this extrinsic as long as the contract
	// won't call `seal_input` in its constructor to copy the data to contract memory.
	// The dummy contract used here does not do this. The costs for the data copy is billed as
	// part of `seal_input`.
	// However, we still use data size as component here as it will be removed by the benchmarking
	// as it has no influence on the weight. This works as "proof" and as regression test.
	call {
		let n in 0 .. u16::max_value() as u32;
		let data = vec![0u8; n as usize];
		let instance = instantiate_contract::<T>(dummy_code(), vec![])?;
		let value = T::Currency::minimum_balance() * 100.into();
		let origin = RawOrigin::Signed(instance.caller.clone());

		// trigger rent collection for worst case performance of call
		System::<T>::set_block_number(eviction_at::<T>(&instance.account_id)? - 5.into());
	}: _(origin, instance.addr, value, Weight::max_value(), data)
	verify {
		// endowment and value transfered via call should be removed from the caller
		assert_eq!(
			T::Currency::free_balance(&instance.caller),
			funding::<T>() - instance.endowment - value,
		);
		// rent should have lowered the amount of balance of the contract
		assert!(T::Currency::free_balance(&instance.account_id) < instance.endowment);
		// but it should not have been evicted by the rent collection
		ensure_alive::<T>(&instance.account_id)?;
	}

	// We benchmark the costs for sucessfully evicting an empty contract.
	// The actual costs are depending on how many storage items the evicted contract
	// does have. However, those costs are not to be payed by the sender but
	// will be distributed over multiple blocks using a scheduler. Otherwise there is
	// no incentive to remove large contracts when the removal is more expensive than
	// the reward for removing them.
	claim_surcharge {
		let instance = instantiate_contract::<T>(dummy_code(), vec![])?;
		let origin = RawOrigin::Signed(instance.caller.clone());
		let account_id = instance.account_id.clone();

		// instantiate should leave us with an alive contract
		ensure_alive::<T>(&instance.account_id)?;

		// generate enough rent so that the contract is evicted
		System::<T>::set_block_number(eviction_at::<T>(&instance.account_id)? + 5.into());
	}: _(origin, account_id, None)
	verify {
		// the claim surcharge should have evicted the contract
		ensure_tombstone::<T>(&instance.account_id)?;

		// the caller should get the reward for being a good snitch
		assert_eq!(
			T::Currency::free_balance(&instance.caller),
			funding::<T>() - instance.endowment + <T as Trait>::SurchargeReward::get(),
		);
	}

	seal_caller {
		let r in 0 .. API_BENCHMARK_BATCHES;
		let instance = instantiate_contract::<T>(getter_code(
			"seal_caller", r * API_BENCHMARK_BATCH_SIZE
		), vec![])?;
		let origin = RawOrigin::Signed(instance.caller.clone());
	}: call(origin, instance.addr, 0.into(), Weight::max_value(), vec![])

	seal_address {
		let r in 0 .. API_BENCHMARK_BATCHES;
		let instance = instantiate_contract::<T>(getter_code(
			"seal_address", r * API_BENCHMARK_BATCH_SIZE
		), vec![])?;
		let origin = RawOrigin::Signed(instance.caller.clone());
	}: call(origin, instance.addr, 0.into(), Weight::max_value(), vec![])

	seal_gas_left {
		let r in 0 .. API_BENCHMARK_BATCHES;
		let instance = instantiate_contract::<T>(getter_code(
			"seal_gas_left", r * API_BENCHMARK_BATCH_SIZE
		), vec![])?;
		let origin = RawOrigin::Signed(instance.caller.clone());
	}: call(origin, instance.addr, 0.into(), Weight::max_value(), vec![])

	seal_balance {
		let r in 0 .. API_BENCHMARK_BATCHES;
		let instance = instantiate_contract::<T>(getter_code(
			"seal_balance", r * API_BENCHMARK_BATCH_SIZE
		), vec![])?;
		let origin = RawOrigin::Signed(instance.caller.clone());
	}: call(origin, instance.addr, 0.into(), Weight::max_value(), vec![])

	seal_value_transferred {
		let r in 0 .. API_BENCHMARK_BATCHES;
		let instance = instantiate_contract::<T>(getter_code(
			"seal_value_transferred", r * API_BENCHMARK_BATCH_SIZE
		), vec![])?;
		let origin = RawOrigin::Signed(instance.caller.clone());
	}: call(origin, instance.addr, 0.into(), Weight::max_value(), vec![])

	seal_minimum_balance {
		let r in 0 .. API_BENCHMARK_BATCHES;
		let instance = instantiate_contract::<T>(getter_code(
			"seal_minimum_balance", r * API_BENCHMARK_BATCH_SIZE
		), vec![])?;
		let origin = RawOrigin::Signed(instance.caller.clone());
	}: call(origin, instance.addr, 0.into(), Weight::max_value(), vec![])

	seal_tombstone_deposit {
		let r in 0 .. API_BENCHMARK_BATCHES;
		let instance = instantiate_contract::<T>(getter_code(
			"seal_tombstone_deposit", r * API_BENCHMARK_BATCH_SIZE
		), vec![])?;
		let origin = RawOrigin::Signed(instance.caller.clone());
	}: call(origin, instance.addr, 0.into(), Weight::max_value(), vec![])

	seal_rent_allowance {
		let r in 0 .. API_BENCHMARK_BATCHES;
		let instance = instantiate_contract::<T>(getter_code(
			"seal_rent_allowance", r * API_BENCHMARK_BATCH_SIZE
		), vec![])?;
		let origin = RawOrigin::Signed(instance.caller.clone());
	}: call(origin, instance.addr, 0.into(), Weight::max_value(), vec![])

	seal_block_number {
		let r in 0 .. API_BENCHMARK_BATCHES;
		let instance = instantiate_contract::<T>(getter_code(
			"seal_block_number", r * API_BENCHMARK_BATCH_SIZE
		), vec![])?;
		let origin = RawOrigin::Signed(instance.caller.clone());
	}: call(origin, instance.addr, 0.into(), Weight::max_value(), vec![])

	seal_gas {
		let r in 0 .. API_BENCHMARK_BATCHES;
		let code = create_code(ModuleDefinition {
			imported_functions: vec![ImportedFunction {
				name: "gas",
				params: vec![ValueType::I32],
				return_type: None,
			}],
			call_body: Some(body_from_repeated(&[
				Instruction::I32Const(42),
				Instruction::Call(0),
			], r * API_BENCHMARK_BATCH_SIZE)),
			.. Default::default()
		});
		let instance = instantiate_contract::<T>(code, vec![])?;
		let origin = RawOrigin::Signed(instance.caller.clone());

	}: call(origin, instance.addr, 0.into(), Weight::max_value(), vec![])

	// We cannot call seal_input multiple times. As a work around we could use the weight of
	// another basic getter for the seal_input base weight and use this benchmark just to
	// determine the overhead for different input sizes.
	seal_input {
		let n in 0 .. max_pages::<T>() * 64;
		let pages = max_pages::<T>();
		let code = create_code::<T>(ModuleDefinition {
			memory: Some(ImportedMemory { min_pages: pages, max_pages: pages }),
			imported_functions: vec![ImportedFunction {
				name: "seal_input",
				params: vec![ValueType::I32, ValueType::I32],
				return_type: None,
			}],
			call_body: Some(body(vec![
				Instruction::I32Const(0), // where to store
				Instruction::I32Const((pages * 64 * 1024 - 4) as i32), // value
				Instruction::I32Store(2, 0), // length gets overwritten
				Instruction::I32Const(4), // ptr where to store output
				Instruction::I32Const(0), // ptr to length
				Instruction::Call(0),
				Instruction::End,
			])),
			.. Default::default()
		});
		let instance = instantiate_contract::<T>(code, vec![])?;
		let data = vec![42u8; (n * 1024).saturating_sub(4) as usize];
		let origin = RawOrigin::Signed(instance.caller.clone());
	}: call(origin, instance.addr, 0.into(), Weight::max_value(), data)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::tests::{ExtBuilder, Test};
	use frame_support::assert_ok;

	#[test]
	fn update_schedule() {
		ExtBuilder::default().build().execute_with(|| {
			assert_ok!(test_benchmark_update_schedule::<Test>());
		});
	}

	#[test]
	fn put_code() {
		ExtBuilder::default().build().execute_with(|| {
			assert_ok!(test_benchmark_put_code::<Test>());
		});
	}

	#[test]
	fn instantiate() {
		ExtBuilder::default().build().execute_with(|| {
			assert_ok!(test_benchmark_instantiate::<Test>());
		});
	}

	#[test]
	fn call() {
		ExtBuilder::default().build().execute_with(|| {
			assert_ok!(test_benchmark_call::<Test>());
		});
	}

	#[test]
	fn claim_surcharge() {
		ExtBuilder::default().build().execute_with(|| {
			assert_ok!(test_benchmark_claim_surcharge::<Test>());
		});
	}

	
	#[test]
	fn seal_caller() {
		ExtBuilder::default().build().execute_with(|| {
			assert_ok!(test_benchmark_seal_caller::<Test>());
		});
	}

	#[test]
	fn seal_address() {
		ExtBuilder::default().build().execute_with(|| {
			assert_ok!(test_benchmark_seal_address::<Test>());
		});
	}

	#[test]
	fn seal_gas_left() {
		ExtBuilder::default().build().execute_with(|| {
			assert_ok!(test_benchmark_seal_gas_left::<Test>());
		});
	}

	#[test]
	fn seal_balance() {
		ExtBuilder::default().build().execute_with(|| {
			assert_ok!(test_benchmark_seal_balance::<Test>());
		});
	}

	#[test]
	fn seal_value_transferred() {
		ExtBuilder::default().build().execute_with(|| {
			assert_ok!(test_benchmark_seal_value_transferred::<Test>());
		});
	}

	#[test]
	fn seal_minimum_balance() {
		ExtBuilder::default().build().execute_with(|| {
			assert_ok!(test_benchmark_seal_minimum_balance::<Test>());
		});
	}

	#[test]
	fn seal_tombstone_deposit() {
		ExtBuilder::default().build().execute_with(|| {
			assert_ok!(test_benchmark_seal_tombstone_deposit::<Test>());
		});
	}

	#[test]
	fn seal_rent_allowance() {
		ExtBuilder::default().build().execute_with(|| {
			assert_ok!(test_benchmark_seal_rent_allowance::<Test>());
		});
	}

	#[test]
	fn seal_block_number() {
		ExtBuilder::default().build().execute_with(|| {
			assert_ok!(test_benchmark_seal_block_number::<Test>());
		});
	}

	#[test]
	fn seal_gas() {
		ExtBuilder::default().build().execute_with(|| {
			assert_ok!(test_benchmark_seal_gas::<Test>());
		});
	}

	#[test]
	fn seal_input() {
		ExtBuilder::default().build().execute_with(|| {
			assert_ok!(test_benchmark_seal_input::<Test>());
		});
	}
}
