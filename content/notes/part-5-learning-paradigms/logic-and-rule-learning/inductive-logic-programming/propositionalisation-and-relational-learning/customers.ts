import type { RelationalDatabase } from 'aifn-compute/logic/induction'

/**
 * Eight customers and their fifteen orders. A customer churned (label 1) exactly when one of their orders was both
 * late and of high value. Amy and Ben have the same number of orders with each attribute value, so only a feature
 * that tests two attributes of the same order tells them apart.
 */
const ORDERS: readonly (readonly [string, string, string, string, string])[] = [
  ['o1', 'amy', 'late', 'high', 'web'],
  ['o2', 'amy', 'on-time', 'low', 'store'],
  ['o3', 'ben', 'late', 'low', 'store'],
  ['o4', 'ben', 'on-time', 'high', 'web'],
  ['o5', 'cat', 'on-time', 'high', 'store'],
  ['o6', 'dan', 'late', 'high', 'store'],
  ['o7', 'eli', 'late', 'low', 'store'],
  ['o8', 'eli', 'on-time', 'low', 'web'],
  ['o9', 'fin', 'on-time', 'high', 'web'],
  ['o10', 'fin', 'late', 'high', 'web'],
  ['o11', 'fin', 'on-time', 'low', 'store'],
  ['o12', 'gia', 'late', 'low', 'web'],
  ['o13', 'gia', 'late', 'low', 'store'],
  ['o14', 'gia', 'on-time', 'high', 'store'],
  ['o15', 'hal', 'on-time', 'low', 'web'],
]

const PEOPLE: readonly (readonly [string, string, number])[] = [
  ['amy', 'basic', 1],
  ['ben', 'basic', 0],
  ['cat', 'basic', 0],
  ['dan', 'premium', 1],
  ['eli', 'basic', 0],
  ['fin', 'basic', 1],
  ['gia', 'premium', 0],
  ['hal', 'basic', 0],
]

export const CUSTOMERS: RelationalDatabase = {
  individuals: PEOPLE.map(([id, plan, label]) => ({ id, attributes: { plan }, label })),
  children: [
    {
      name: 'order',
      rows: ORDERS.map(([id, parent, delivery, value, channel]) => ({
        id,
        parent,
        attributes: { delivery, value, channel },
      })),
    },
  ],
}
